// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Greedy decode benchmark for the Bonsai-27B (qwen3_5) export converted with
// SDPAToPagedAttention. Versus the stateful path this replaces the 9-op
// causal-conv1d chain with one PagedCausalConv1D per linear-attention layer and
// the Loop-derived GatedDeltaNet with PagedGatedDeltaNet.
//
// BENCH_MTP=<openvino_mtp_model.xml> (from bonsai2_mtp_to_ir.py) turns on MTP
// speculative decoding with BENCH_MTP_K draft tokens per step (default 3):
// the draft proposes k tokens, the target verifies [last, d1..dk] in one
// M = k+1 step, and the linear-attention state is rolled back to the last
// accepted token through la.cache_interval = 1 (the GDN and conv kernels then
// write the state after every token into its own la.block_indices slot).
// Greedy acceptance, so the output matches the non-speculative run up to
// near-tie numerics.

#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>

#include "paged_llm_27b_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kFullAttnLayers = 16;
constexpr int32_t kLinearAttnLayers = 48;
constexpr int32_t kKVHeads = 4;
constexpr int32_t kHeadDim = 256;
constexpr int32_t kConvChannels = 10240;
constexpr int32_t kConvKernel = 4;
constexpr int32_t kGdnValueHeads = 48;
constexpr int32_t kGdnKeyDim = 128;
constexpr int32_t kGdnValueDim = 128;
constexpr int32_t kHidden = 5120;

std::vector<int64_t> parse_ids(const std::string& csv) {
    std::vector<int64_t> ids;
    std::stringstream stream(csv);
    std::string value;
    while (std::getline(stream, value, ','))
        if (!value.empty())
            ids.push_back(std::stoll(value));
    return ids;
}

void set_i32(ov::InferRequest& request, const std::string& name, const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    request.set_tensor(name, tensor);
}

void set_i32_scalar(ov::InferRequest& request, const std::string& name, int32_t value) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{});
    *tensor.data<int32_t>() = value;
    request.set_tensor(name, tensor);
}

// Full-attention KV blocks; the linear-attention state is per-sequence and uses
// a single slot, so it is not paged the same way.
void set_active_blocks(ov::InferRequest& request, int32_t token_count, int32_t block_size) {
    const int32_t active = (token_count + block_size - 1) / block_size;
    std::vector<int32_t> block_indices(active);
    for (int32_t i = 0; i < active; ++i)
        block_indices[i] = i;
    set_i32(request, "block_indices", block_indices);
    set_i32(request, "block_indices_begins", {0, active});
}

// The kernel reads the state from block_indices[begin] and writes the updated
// state to block_indices[begin + 1], so the two slots must swap every call.
// cache_interval <= 0 keeps that to a single write per call.
constexpr int32_t kLinearAttnSlots = 2;

void set_linear_attn_meta(ov::InferRequest& request, int32_t past_length, int32_t read_slot) {
    set_i32(request, "la.past_lens", {past_length});
    set_i32(request, "la.block_indices", {read_slot, 1 - read_slot});
    set_i32(request, "la.block_indices_begins", {0, kLinearAttnSlots});
    set_i32(request, "la.cache_interval", {0});
}

// Speculative verify: cache_interval 1 writes the state after token t of the
// step to slots[1 + t], so any accepted prefix can be resumed from.
void set_linear_attn_meta_spec(ov::InferRequest& request, int32_t past_length, const std::vector<int32_t>& slots) {
    set_i32(request, "la.past_lens", {past_length});
    set_i32(request, "la.block_indices", slots);
    set_i32(request, "la.block_indices_begins", {0, static_cast<int32_t>(slots.size())});
    set_i32(request, "la.cache_interval", {1});
}

size_t logit_rows(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    size_t rows = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
        rows *= shape[i];
    return rows;
}

int64_t argmax_row(const ov::Tensor& logits, size_t r) {
    const size_t vocab = logits.get_shape().back();
    const float* row = logits.data<float>() + r * vocab;
    return static_cast<int64_t>(std::distance(row, std::max_element(row, row + vocab)));
}

int64_t argmax_last(const ov::Tensor& logits) {
    return argmax_row(logits, logit_rows(logits) - 1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: paged_bench_llm_27b <paged_language_model.xml> <text_embeddings.xml>"
                     " <device> [max_new_tokens] [comma-separated input ids]\n";
        return 1;
    }
    const std::string lm_path = argv[1];
    const std::string embed_path = argv[2];
    const std::string device = argv[3];
    const int max_new = argc > 4 ? std::stoi(argv[4]) : 256;
    const auto prompt = parse_ids(argc > 5 ? argv[5] : "151644,872,198");
    if (prompt.empty()) {
        std::cerr << "empty prompt\n";
        return 1;
    }

    const int32_t block_size = std::getenv("BENCH_BLOCK_SIZE") ? std::stoi(std::getenv("BENCH_BLOCK_SIZE")) : 16;
    const int32_t max_context_len =
        std::getenv("BENCH_MAX_LEN") ? std::stoi(std::getenv("BENCH_MAX_LEN")) : 512;
    const int32_t num_blocks = (max_context_len + block_size - 1) / block_size;
    const char* mtp_path = std::getenv("BENCH_MTP");
    const int32_t mtp_k = mtp_path ? (std::getenv("BENCH_MTP_K") ? std::stoi(std::getenv("BENCH_MTP_K")) : 3) : 0;
    const int32_t la_slots = mtp_path ? mtp_k + 2 : kLinearAttnSlots;
    if (static_cast<int32_t>(prompt.size()) + max_new + mtp_k + 1 > max_context_len) {
        std::cerr << "prompt plus generated tokens must fit " << max_context_len << " tokens\n";
        return 1;
    }

    try {
        ov::Core core;
        const char* precision = std::getenv("BENCH_PRECISION");
        ov::AnyMap config{{"INFERENCE_PRECISION_HINT", precision ? precision : "f16"}};
        if (std::getenv("BENCH_PROFILE"))
            config[ov::enable_profiling.name()] = true;
        if (const char* cache = std::getenv("OV_CACHE_DIR"))
            core.set_property(ov::cache_dir(cache));

        const auto compile_start = std::chrono::high_resolution_clock::now();
        auto embed_model = core.read_model(embed_path);
        embed_model->reshape({{embed_model->input(0).get_any_name(), ov::PartialShape{1, -1}}});
        auto embed = core.compile_model(embed_model, device, config);
        // Converted in-process: the paged ops are extension-opset and cannot be
        // deserialized back from a saved IR.
        auto lm_model = core.read_model(lm_path);
        if (mtp_path)  // the draft consumes the target's post-final-norm hidden state
            paged27b::add_output(lm_model, paged27b::find_by_suffix(lm_model, paged27b::kFinalNorm)->output(0),
                                 "last_hidden_state");
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        manager.run_passes(lm_model);
        auto lm = core.compile_model(lm_model, device, config);
        ov::CompiledModel draft;
        if (mtp_path) {
            auto draft_model = core.read_model(mtp_path);
            ov::pass::Manager draft_manager;
            draft_manager.register_pass<ov::pass::SDPAToPagedAttention>();
            draft_manager.run_passes(draft_model);
            paged27b::add_row_gather(draft_model, paged27b::find_by_suffix(draft_model, paged27b::kDraftLayerOut));
            draft = core.compile_model(draft_model, device, config);
        }
        const auto compile_end = std::chrono::high_resolution_clock::now();
        std::cout << "compile " << std::chrono::duration<double>(compile_end - compile_start).count() << " s\n";

        if (std::getenv("BENCH_RTOPS")) {
            std::map<std::string, size_t> by_type;
            for (const auto& node : lm.get_runtime_model()->get_ordered_ops()) {
                const auto& rt = node->get_rt_info();
                const auto it = rt.find("layerType");
                by_type[it != rt.end() ? it->second.as<std::string>() : node->get_type_name()]++;
            }
            size_t total = 0;
            for (const auto& e : by_type)
                total += e.second;
            std::vector<std::pair<std::string, size_t>> v(by_type.begin(), by_type.end());
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::cout << "runtime graph: " << total << " nodes\n";
            for (size_t i = 0; i < v.size() && i < 24; ++i)
                std::cout << "   " << std::setw(30) << std::left << v[i].first << v[i].second << '\n';
        }

        auto embed_request = embed.create_infer_request();
        auto request = lm.create_infer_request();

        // Held for the whole run: the kernels update these in place. Shapes and
        // precisions are decided by the plugin's ConvertPagedAttnInputs pass, so
        // take them from the compiled model and only fill in the block count.
        // They must be device tensors: a host tensor is re-uploaded before every
        // infer, which would discard the in-place state update.
        std::vector<ov::Tensor> caches;
        const bool host_caches = std::getenv("BENCH_HOST_CACHES") != nullptr;
        ov::RemoteContext context = lm.get_context();
        auto bind = [&](const std::string& name, size_t blocks) {
            const auto port = lm.input(name);
            const auto partial = port.get_partial_shape();
            ov::Shape shape(partial.size());
            for (size_t i = 0; i < partial.size(); ++i)
                shape[i] = partial[i].is_dynamic() ? blocks : static_cast<size_t>(partial[i].get_length());
            ov::Tensor zeros(port.get_element_type(), shape);
            std::memset(zeros.data(), 0, zeros.get_byte_size());
            if (host_caches) {
                caches.emplace_back(std::move(zeros));
            } else {
                auto remote = context.create_tensor(port.get_element_type(), shape);
                remote.copy_from(zeros);
                caches.emplace_back(std::move(remote));
            }
            request.set_tensor(name, caches.back());
        };
        for (int32_t layer = 0; layer < kFullAttnLayers; ++layer) {
            bind("key_cache." + std::to_string(layer), static_cast<size_t>(num_blocks));
            bind("value_cache." + std::to_string(layer), static_cast<size_t>(num_blocks));
        }
        for (int32_t layer = 0; layer < kLinearAttnLayers; ++layer) {
            bind("conv_state_table." + std::to_string(layer), static_cast<size_t>(la_slots));
            bind("gated_delta_state_table." + std::to_string(layer), static_cast<size_t>(la_slots));
        }
        set_i32_scalar(request, "max_context_len", max_context_len);
        ov::InferRequest draft_request;
        if (mtp_path) {
            draft_request = draft.create_infer_request();
            ov::RemoteContext draft_context = draft.get_context();
            for (const auto& name : {"key_cache.0", "value_cache.0"}) {
                const auto port = draft.input(name);
                const auto partial = port.get_partial_shape();
                ov::Shape shape(partial.size());
                for (size_t i = 0; i < partial.size(); ++i)
                    shape[i] = partial[i].is_dynamic() ? static_cast<size_t>(num_blocks)
                                                       : static_cast<size_t>(partial[i].get_length());
                ov::Tensor zeros(port.get_element_type(), shape);
                std::memset(zeros.data(), 0, zeros.get_byte_size());
                auto remote = draft_context.create_tensor(port.get_element_type(), shape);
                remote.copy_from(zeros);
                caches.emplace_back(std::move(remote));
                draft_request.set_tensor(name, caches.back());
            }
            set_i32_scalar(draft_request, "max_context_len", max_context_len);
        }
        if (std::getenv("BENCH_MEMINFO")) {
            size_t bytes = 0;
            for (const auto& t : caches)
                bytes += t.get_byte_size();
            std::cout << "caches: " << caches.size() << " tensors, " << bytes / (1024 * 1024) << " MiB, "
                      << (host_caches ? "host" : "device") << '\n';
        }

        // inputs_embeds is [tokens, hidden] on the paged path, not [1, tokens, hidden].
        auto embed_host = [&](const std::vector<int64_t>& ids) {
            ov::Tensor input(ov::element::i64, ov::Shape{1, ids.size()});
            std::copy(ids.begin(), ids.end(), input.data<int64_t>());
            embed_request.set_input_tensor(input);
            embed_request.infer();
            return embed_request.get_output_tensor();
        };
        auto feed_embeds = [&](const std::vector<int64_t>& ids) {
            const auto out = embed_host(ids);
            ov::Tensor embeds(lm.input("inputs_embeds").get_element_type(), ov::Shape{ids.size(), kHidden});
            std::memcpy(embeds.data(), out.data(), embeds.get_byte_size());
            request.set_tensor("inputs_embeds", embeds);
        };
        auto set_positions_on = [&](ov::InferRequest& req, int64_t start, size_t length) {
            ov::Tensor positions(ov::element::i64, ov::Shape{4, length});
            auto* data = positions.data<int64_t>();
            for (size_t section = 0; section < 4; ++section)
                for (size_t i = 0; i < length; ++i)
                    data[section * length + i] = start + static_cast<int64_t>(i);
            req.set_tensor("position_ids", positions);
        };
        auto set_positions = [&](int64_t start, size_t length) {
            set_positions_on(request, start, length);
        };

        // MTP draft step over n rows [embed(ids[j]) | hid[j]] at positions pos..pos+n-1
        // (its KV cache holds pos valid entries); returns the draft token after the
        // last row and leaves that row's hidden state in draft_hidden.
        std::vector<float> draft_hidden(kHidden);
        double tm_embed = 0, tm_dinfer = 0, tm_dout = 0, tm_tinfer = 0, tm_tout = 0;
        auto ms_since = [](std::chrono::high_resolution_clock::time_point t) {
            return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t).count();
        };
        auto draft_run = [&](const std::vector<int64_t>& ids, const float* hid, int32_t pos) {
            const size_t n = ids.size();
            auto tt = std::chrono::high_resolution_clock::now();
            const auto emb = embed_host(ids);
            tm_embed += ms_since(tt);
            if (emb.get_element_type() != ov::element::f32)
                throw std::runtime_error("MTP: expected f32 embeddings");
            ov::Tensor x(ov::element::f32, ov::Shape{n, 2 * static_cast<size_t>(kHidden)});
            float* xd = x.data<float>();
            const float* ed = emb.data<float>();
            for (size_t j = 0; j < n; ++j) {
                std::memcpy(xd + j * 2 * kHidden, ed + j * kHidden, kHidden * sizeof(float));
                std::memcpy(xd + j * 2 * kHidden + kHidden, hid + j * kHidden, kHidden * sizeof(float));
            }
            draft_request.set_tensor("inputs_embeds", x);
            set_positions_on(draft_request, pos, n);
            set_i32(draft_request, "past_lens", {pos});
            set_i32(draft_request, "subsequence_begins", {0, static_cast<int32_t>(n)});
            set_active_blocks(draft_request, pos + static_cast<int32_t>(n), block_size);
            set_i32(draft_request, "logit_rows", {static_cast<int32_t>(n) - 1});
            tt = std::chrono::high_resolution_clock::now();
            draft_request.infer();
            tm_dinfer += ms_since(tt);
            tt = std::chrono::high_resolution_clock::now();
            const auto h = draft_request.get_tensor("last_hidden_state");
            std::memcpy(draft_hidden.data(), h.data<float>() + (h.get_size() - kHidden), kHidden * sizeof(float));
            const int64_t tok = argmax_last(draft_request.get_tensor("logits"));
            tm_dout += ms_since(tt);
            return tok;
        };
        std::vector<float> prompt_hidden;
        if (mtp_path)
            prompt_hidden.resize(prompt.size() * kHidden);
        auto copy_hidden = [&](float* dst, size_t rows) {
            const auto h = request.get_tensor("last_hidden_state");
            if (h.get_element_type() != ov::element::f32 || h.get_size() < rows * kHidden)
                throw std::runtime_error("MTP: unexpected last_hidden_state");
            std::memcpy(dst, h.data<float>(), rows * kHidden * sizeof(float));
        };

        const auto prefill_start = std::chrono::high_resolution_clock::now();
        int32_t read_slot = 0;
        // A whole-prompt prefill needs an ESIMD scratch surface that does not
        // fit next to the weights on an integrated GPU; chunking bounds it.
        const int32_t chunk = std::getenv("BENCH_PREFILL_CHUNK")
                                  ? std::stoi(std::getenv("BENCH_PREFILL_CHUNK"))
                                  : static_cast<int32_t>(prompt.size());
        for (size_t start = 0; start < prompt.size(); start += static_cast<size_t>(chunk)) {
            const size_t len = std::min(static_cast<size_t>(chunk), prompt.size() - start);
            const std::vector<int64_t> part(prompt.begin() + static_cast<long>(start),
                                            prompt.begin() + static_cast<long>(start + len));
            feed_embeds(part);
            set_positions(static_cast<int64_t>(start), len);
            set_i32(request, "past_lens", {static_cast<int32_t>(start)});
            set_i32(request, "subsequence_begins", {0, static_cast<int32_t>(len)});
            set_active_blocks(request, static_cast<int32_t>(start + len), block_size);
            set_linear_attn_meta(request, static_cast<int32_t>(start), read_slot);
            request.infer();
            read_slot = 1 - read_slot;
            if (mtp_path)
                copy_hidden(prompt_hidden.data() + start * kHidden, len);
        }
        const auto prefill_end = std::chrono::high_resolution_clock::now();

        int64_t next = argmax_last(request.get_tensor("logits"));
        std::vector<int64_t> generated{next};

        const bool no_eos = std::getenv("BENCH_NO_EOS") != nullptr;
        auto is_eos = [&](int64_t t) {
            return !no_eos && (t == 248046 || t == 248044);
        };
        int decoded = 0;
        const auto decode_start = std::chrono::high_resolution_clock::now();
        int64_t mtp_rounds = 0, mtp_accepted = 0;
        double t_verify = 0, t_draft = 0;
        using clk = std::chrono::high_resolution_clock;
        for (int step = 1; !mtp_path && step < max_new; ++step) {
            const int32_t past = static_cast<int32_t>(prompt.size()) + step - 1;
            feed_embeds({next});
            set_positions(past, 1);
            set_i32(request, "past_lens", {past});
            set_i32(request, "subsequence_begins", {0, 1});
            set_active_blocks(request, past + 1, block_size);
            set_linear_attn_meta(request, past, read_slot);
            request.infer();
            read_slot = 1 - read_slot;
            next = argmax_last(request.get_tensor("logits"));
            generated.push_back(next);
            ++decoded;
            if (is_eos(next))
                break;
        }
        if (mtp_path && !is_eos(next) && max_new > 1) {
            const int32_t P = static_cast<int32_t>(prompt.size());
            const bool debug = std::getenv("BENCH_MTP_DEBUG") != nullptr;
            // draft prefill: row j pairs the target hidden state at j with token j+1
            auto t0 = clk::now();
            std::vector<int64_t> shifted(prompt.begin() + 1, prompt.end());
            shifted.push_back(next);
            int64_t d = 0;
            for (size_t start = 0; start < shifted.size(); start += static_cast<size_t>(chunk)) {
                const size_t len = std::min(static_cast<size_t>(chunk), shifted.size() - start);
                d = draft_run(std::vector<int64_t>(shifted.begin() + static_cast<long>(start),
                                                   shifted.begin() + static_cast<long>(start + len)),
                              prompt_hidden.data() + start * kHidden,
                              static_cast<int32_t>(start));
            }
            std::vector<int64_t> drafts{d};
            for (int32_t i = 1; i < mtp_k; ++i)
                drafts.push_back(d = draft_run({d}, draft_hidden.data(), P - 1 + i));
            t_draft += std::chrono::duration<double, std::milli>(clk::now() - t0).count();

            int32_t past = P;
            int64_t last = next;
            std::vector<float> verify_hidden(static_cast<size_t>(mtp_k + 1) * kHidden);
            bool stop = false;
            while (!stop) {
                t0 = clk::now();
                std::vector<int64_t> tokens{last};
                tokens.insert(tokens.end(), drafts.begin(), drafts.end());
                const int32_t m = static_cast<int32_t>(tokens.size());
                auto tt = clk::now();
                feed_embeds(tokens);
                tm_embed += ms_since(tt);
                set_positions(past, tokens.size());
                set_i32(request, "past_lens", {past});
                set_i32(request, "subsequence_begins", {0, m});
                set_active_blocks(request, past + m, block_size);
                std::vector<int32_t> slots{read_slot};
                for (int32_t s = 0; s < la_slots; ++s)
                    if (s != read_slot)
                        slots.push_back(s);
                set_linear_attn_meta_spec(request, past, slots);
                tt = clk::now();
                request.infer();
                tm_tinfer += ms_since(tt);
                tt = clk::now();
                const auto logits = request.get_tensor("logits");
                std::vector<int64_t> target(tokens.size());
                for (size_t i = 0; i < tokens.size(); ++i)
                    target[i] = argmax_row(logits, i);
                tm_tout += ms_since(tt);
                int32_t a = 0;
                while (a < mtp_k && drafts[a] == target[a])
                    ++a;
                t_verify += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                if (debug) {
                    std::cout << "  round " << mtp_rounds << " past " << past << " drafts";
                    for (auto t : drafts)
                        std::cout << ' ' << t;
                    std::cout << " target";
                    for (auto t : target)
                        std::cout << ' ' << t;
                    std::cout << " accepted " << a << '\n';
                }
                ++mtp_rounds;
                mtp_accepted += a;
                for (int32_t i = 0; i <= a && !stop; ++i) {
                    const int64_t t = i < a ? drafts[i] : target[a];
                    generated.push_back(t);
                    stop = is_eos(t) || static_cast<int>(generated.size()) >= max_new;
                }
                read_slot = slots[1 + a];
                if (stop)
                    break;

                t0 = clk::now();
                copy_hidden(verify_hidden.data(), static_cast<size_t>(a + 1));
                std::vector<int64_t> ids(drafts.begin(), drafts.begin() + a);
                ids.push_back(target[a]);
                d = draft_run(ids, verify_hidden.data(), past);
                drafts.assign(1, d);
                for (int32_t i = 1; i < mtp_k; ++i)
                    drafts.push_back(d = draft_run({d}, draft_hidden.data(), past + a + i));
                t_draft += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                past += a + 1;
                last = target[a];
            }
            decoded = static_cast<int>(generated.size()) - 1;
        }
        const auto decode_end = std::chrono::high_resolution_clock::now();
        const double decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();

        std::cout << "TTFT (prefill) : "
                  << std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count() << " ms\n";
        std::cout << "decode         : " << decoded << " tokens in " << decode_seconds
                  << " s = " << decoded / decode_seconds << " tok/s\n";
        if (mtp_path && mtp_rounds > 0)
            std::cout << "mtp            : k=" << mtp_k << ", " << mtp_rounds << " rounds, acceptance "
                      << 100.0 * static_cast<double>(mtp_accepted) / static_cast<double>(mtp_rounds * mtp_k)
                      << "%, " << static_cast<double>(mtp_accepted + mtp_rounds) / static_cast<double>(mtp_rounds)
                      << " tokens/round, verify " << t_verify / static_cast<double>(mtp_rounds)
                      << " ms/round, draft " << t_draft / static_cast<double>(mtp_rounds) << " ms/round\n"
                      << "mtp host split : embed " << tm_embed / mtp_rounds << ", target infer " << tm_tinfer / mtp_rounds
                      << ", target out " << tm_tout / mtp_rounds << ", draft infer " << tm_dinfer / mtp_rounds
                      << ", draft out " << tm_dout / mtp_rounds << " ms/round\n";
        std::cout << "generated_ids  =";
        for (const auto id : generated)
            std::cout << id << ',';
        std::cout << '\n';

        auto dump_profile = [](ov::InferRequest& req, const char* label) {
            std::map<std::string, std::pair<double, size_t>> by_type;
            double total = 0;
            const bool by_fc = std::getenv("BENCH_PROFILE_FC") != nullptr;
            for (const auto& p : req.get_profiling_info()) {
                if (p.status == ov::ProfilingInfo::Status::NOT_RUN)
                    continue;
                const double us = static_cast<double>(p.real_time.count());
                std::string key = p.node_type;
                if (by_fc && key.rfind("FullyConnected", 0) == 0) {
                    // layers.<i>.<module>.<proj>/... -> <proj>
                    const auto slash = p.node_name.find('/');
                    std::string head = p.node_name.substr(0, slash);
                    const auto dot = head.rfind('.');
                    key += ":" + (dot == std::string::npos ? head : head.substr(dot + 1));
                }
                auto& e = by_type[key];
                e.first += us;
                e.second += 1;
                total += us;
            }
            std::vector<std::pair<std::string, std::pair<double, size_t>>> v(by_type.begin(), by_type.end());
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
            std::cout << label << " (total " << total << " us):\n";
            for (size_t i = 0; i < v.size() && i < 20; ++i)
                std::cout << "  " << std::setw(28) << std::left << v[i].first << std::setw(10) << std::right
                          << v[i].second.first << " us  n=" << v[i].second.second << "  ("
                          << (100.0 * v[i].second.first / total) << "%)\n";
        };
        if (std::getenv("BENCH_PROFILE")) {
            dump_profile(request, mtp_path ? "last verify step profile" : "per-token profile");
            if (mtp_path)
                dump_profile(draft_request, "last draft step profile");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "paged 27b decode failed: " << error.what() << '\n';
        return 1;
    }
}
