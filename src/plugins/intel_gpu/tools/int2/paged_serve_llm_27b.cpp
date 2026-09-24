// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Batched greedy generation for the 27B (qwen3_5) paged export, for running an
// evaluation harness against the GPU plugin. Requests are read from a file and
// served with continuous batching: up to `batch` sequences decode together, and
// a finished slot is refilled (one prefill, alone) from the pending queue.
//
//   paged_serve_llm_27b <lm.xml> <embeddings.xml> <device> <requests> <outputs> [batch]
//
// <requests>: one line per request, `max_new_tokens;id,id,id,...`
// <outputs> : one line per request in the same order, the generated ids
//             (including the EOS that stopped it, if any).
// Env: BENCH_MAX_LEN (per-sequence context, default 6144), BENCH_PRECISION,
//      BENCH_EOS (comma list, default 248044,248046), BENCH_BLOCK_SIZE,
//      BENCH_VERBOSE=1 (per-request progress on stderr),
//      BENCH_MTP=<openvino_mtp_model.xml> + BENCH_MTP_K (default 3): MTP
//      speculative decoding. Each round runs the draft over all active slots
//      (first step: the rows the last verify accepted, or the whole prompt for
//      a new request; then k-1 single-row steps), then one joint verify of
//      [next, d1..dk] per slot with la.cache_interval = 1, so every slot resumes
//      from the linear-attention state after its last accepted token.

#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
// dev API: ROI copy into a remote tensor (zeroing one slot of a state table)
#include <openvino/runtime/iremote_tensor.hpp>
#include <openvino/runtime/make_tensor.hpp>

#include "paged_llm_27b_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kFullAttnLayers = 16;
constexpr int32_t kLinearAttnLayers = 48;
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

struct Request {
    int max_new = 256;
    std::vector<int64_t> prompt;
};

struct Slot {
    bool active = false;
    size_t req = 0;
    int32_t past = 0;        // tokens already in the caches
    int32_t read_slot = 0;   // linear-attention state slot to read this step
    int64_t next = 0;        // token to feed this step
    std::vector<int64_t> generated;
    // MTP: rows for the next draft step, from position draft_past on
    std::vector<int64_t> pend_ids;
    std::vector<float> pend_hid;
    int32_t draft_past = 0;
    std::vector<int64_t> drafts;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: paged_serve_llm_27b <lm.xml> <embeddings.xml> <device> <requests> <outputs> [batch]\n";
        return 1;
    }
    const std::string lm_path = argv[1], embed_path = argv[2], device = argv[3];
    const std::string req_path = argv[4], out_path = argv[5];
    const int32_t batch = argc > 6 ? std::stoi(argv[6]) : 8;
    const int32_t block_size = std::getenv("BENCH_BLOCK_SIZE") ? std::stoi(std::getenv("BENCH_BLOCK_SIZE")) : 16;
    const int32_t max_context_len = std::getenv("BENCH_MAX_LEN") ? std::stoi(std::getenv("BENCH_MAX_LEN")) : 6144;
    const int32_t blocks_per_seq = (max_context_len + block_size - 1) / block_size;
    const auto eos = parse_ids(std::getenv("BENCH_EOS") ? std::getenv("BENCH_EOS") : "248044,248046");
    const bool verbose = std::getenv("BENCH_VERBOSE") != nullptr;
    const char* mtp_path = std::getenv("BENCH_MTP");
    const int32_t mtp_k = mtp_path ? (std::getenv("BENCH_MTP_K") ? std::stoi(std::getenv("BENCH_MTP_K")) : 3) : 0;
    // linear-attention state slots per sequence: read + write, or read + one
    // per verified token
    const int32_t la_slots = mtp_path ? mtp_k + 2 : 2;

    std::vector<Request> requests;
    {
        std::ifstream in(req_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty())
                continue;
            const auto semi = line.find(';');
            Request r;
            r.max_new = std::stoi(line.substr(0, semi));
            r.prompt = parse_ids(line.substr(semi + 1));
            if (r.prompt.empty() || static_cast<int32_t>(r.prompt.size()) + r.max_new + mtp_k + 1 > max_context_len) {
                std::cerr << "request " << requests.size() << ": " << r.prompt.size() << "+" << r.max_new
                          << " tokens do not fit BENCH_MAX_LEN=" << max_context_len << "\n";
                return 1;
            }
            requests.push_back(std::move(r));
        }
    }
    std::vector<std::vector<int64_t>> outputs(requests.size());
    std::cerr << requests.size() << " requests, batch " << batch << ", context " << max_context_len << "\n";

    try {
        ov::Core core;
        const char* precision = std::getenv("BENCH_PRECISION");
        ov::AnyMap config{{"INFERENCE_PRECISION_HINT", precision ? precision : "f16"}};
        if (const char* cache = std::getenv("OV_CACHE_DIR"))
            core.set_property(ov::cache_dir(cache));

        const auto t_compile = std::chrono::high_resolution_clock::now();
        auto embed_model = core.read_model(embed_path);
        embed_model->reshape({{embed_model->input(0).get_any_name(), ov::PartialShape{1, -1}}});
        auto embed = core.compile_model(embed_model, device, config);
        auto lm_model = core.read_model(lm_path);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        manager.run_passes(lm_model);
        // Only the last token of a prefill needs logits, but the graph computes
        // them for every row (1260 x 248320 fp32 = 1.25 GB copied back). Gather
        // the wanted rows in front of the head; decode asks for every row.
        {
            const auto head_in = paged27b::find_by_suffix(lm_model, paged27b::kFinalNorm);
            paged27b::add_row_gather(lm_model, head_in);
            if (mtp_path)  // every row, for the draft
                paged27b::add_output(lm_model, head_in->output(0), "last_hidden_state");
        }
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
        std::cerr << "compile " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_compile).count()
                  << " s\n";

        auto embed_request = embed.create_infer_request();
        auto request = lm.create_infer_request();

        std::vector<ov::Tensor> caches;
        std::vector<std::string> cache_names;
        ov::RemoteContext context = lm.get_context();
        auto bind = [&](const std::string& name, size_t blocks) {
            const auto port = lm.input(name);
            const auto partial = port.get_partial_shape();
            ov::Shape shape(partial.size());
            for (size_t i = 0; i < partial.size(); ++i)
                shape[i] = partial[i].is_dynamic() ? blocks : static_cast<size_t>(partial[i].get_length());
            ov::Tensor zeros(port.get_element_type(), shape);
            std::memset(zeros.data(), 0, zeros.get_byte_size());
            auto remote = context.create_tensor(port.get_element_type(), shape);
            remote.copy_from(zeros);
            caches.emplace_back(std::move(remote));
            cache_names.push_back(name);
            request.set_tensor(name, caches.back());
        };
        for (int32_t layer = 0; layer < kFullAttnLayers; ++layer) {
            bind("key_cache." + std::to_string(layer), static_cast<size_t>(blocks_per_seq) * batch);
            bind("value_cache." + std::to_string(layer), static_cast<size_t>(blocks_per_seq) * batch);
        }
        for (int32_t layer = 0; layer < kLinearAttnLayers; ++layer) {
            bind("conv_state_table." + std::to_string(layer), static_cast<size_t>(la_slots) * batch);
            bind("gated_delta_state_table." + std::to_string(layer), static_cast<size_t>(la_slots) * batch);
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
                    shape[i] = partial[i].is_dynamic() ? static_cast<size_t>(blocks_per_seq) * batch
                                                       : static_cast<size_t>(partial[i].get_length());
                ov::Tensor zeros(port.get_element_type(), shape);
                std::memset(zeros.data(), 0, zeros.get_byte_size());
                auto remote = draft_context.create_tensor(port.get_element_type(), shape);
                remote.copy_from(zeros);
                caches.emplace_back(std::move(remote));
                cache_names.push_back(std::string("draft.") + name);
                draft_request.set_tensor(name, caches.back());
            }
            set_i32_scalar(draft_request, "max_context_len", max_context_len);
        }
        // One zeroed device tensor per state-table shape ([la_slots, ...]);
        // zero_slot_state copies it over a sequence's slots device-to-device.
        std::vector<size_t> la_cache_idx;
        std::map<std::string, ov::Tensor> zero_blocks;
        for (size_t i = 0; i < caches.size(); ++i) {
            if (cache_names[i].rfind("conv_state_table", 0) != 0 && cache_names[i].rfind("gated_delta_state_table", 0) != 0)
                continue;
            la_cache_idx.push_back(i);
            ov::Shape slot_shape = caches[i].get_shape();
            slot_shape[0] = la_slots;
            const std::string key = caches[i].get_element_type().get_type_name() + ":" + slot_shape.to_string();
            if (!zero_blocks.count(key)) {
                ov::Tensor zeros(caches[i].get_element_type(), slot_shape);
                std::memset(zeros.data(), 0, zeros.get_byte_size());
                auto remote = context.create_tensor(caches[i].get_element_type(), slot_shape);
                remote.copy_from(zeros);
                zero_blocks.emplace(key, std::move(remote));
            }
        }
        auto zero_slot_state = [&](int32_t s) {
            for (const size_t i : la_cache_idx) {
                ov::Shape slot_shape = caches[i].get_shape();
                slot_shape[0] = la_slots;
                const std::string key = caches[i].get_element_type().get_type_name() + ":" + slot_shape.to_string();
                const size_t slot_bytes = caches[i].get_byte_size() / caches[i].get_shape()[0];
                auto dst = std::dynamic_pointer_cast<ov::IRemoteTensor>(ov::get_tensor_impl(caches[i])._ptr);
                auto src = ov::get_tensor_impl(zero_blocks.at(key))._ptr;
                if (!dst)
                    throw std::runtime_error("state table is not a remote tensor");
                dst->copy_from(src, 0, static_cast<size_t>(s) * la_slots * slot_bytes, slot_shape);
            }
        };
        {
            size_t bytes = 0;
            for (const auto& t : caches)
                bytes += t.get_byte_size();
            std::cerr << "caches: " << bytes / (1024 * 1024) << " MiB\n";
        }

        const auto embeds_type = lm.input("inputs_embeds").get_element_type();
        auto feed_embeds = [&](const std::vector<int64_t>& ids) {
            ov::Tensor input(ov::element::i64, ov::Shape{1, ids.size()});
            std::copy(ids.begin(), ids.end(), input.data<int64_t>());
            embed_request.set_input_tensor(input);
            embed_request.infer();
            const auto out = embed_request.get_output_tensor();
            ov::Tensor embeds(embeds_type, ov::Shape{ids.size(), kHidden});
            std::memcpy(embeds.data(), out.data(), embeds.get_byte_size());
            request.set_tensor("inputs_embeds", embeds);
        };

        // One infer over the given (slot, token count) pairs; tokens of each
        // slot start at its `past`. Fills the paged-attention and linear-
        // attention metadata for every subsequence in the batch. `spec` (MTP
        // verify): every row gets logits, the state after each token goes to
        // its own slot, and the caller advances past/read_slot by what it
        // accepted.
        std::vector<Slot> slots(batch);
        auto run_step = [&](const std::vector<std::pair<int32_t, int32_t>>& parts, bool spec = false) {
            size_t total = 0;
            for (const auto& p : parts)
                total += static_cast<size_t>(p.second);
            ov::Tensor positions(ov::element::i64, ov::Shape{4, total});
            auto* pos = positions.data<int64_t>();
            std::vector<int32_t> past_lens, subseq{0}, blocks, block_begins{0}, la_past, la_blocks, la_begins{0};
            size_t tok = 0;
            for (const auto& [s, len] : parts) {
                const Slot& sl = slots[s];
                for (int32_t i = 0; i < len; ++i, ++tok)
                    for (size_t section = 0; section < 4; ++section)
                        pos[section * total + tok] = sl.past + i;
                past_lens.push_back(sl.past);
                subseq.push_back(subseq.back() + len);
                const int32_t active = (sl.past + len + block_size - 1) / block_size;
                for (int32_t b = 0; b < active; ++b)
                    blocks.push_back(s * blocks_per_seq + b);
                block_begins.push_back(block_begins.back() + active);
                la_past.push_back(sl.past);
                la_blocks.push_back(s * la_slots + sl.read_slot);
                if (spec) {
                    for (int32_t o = 0; o < la_slots; ++o)
                        if (o != sl.read_slot)
                            la_blocks.push_back(s * la_slots + o);
                    la_begins.push_back(la_begins.back() + la_slots);
                } else {
                    la_blocks.push_back(s * la_slots + (1 - sl.read_slot));
                    la_begins.push_back(la_begins.back() + 2);
                }
            }
            request.set_tensor("position_ids", positions);
            set_i32(request, "past_lens", past_lens);
            set_i32(request, "subsequence_begins", subseq);
            set_i32(request, "block_indices", blocks);
            set_i32(request, "block_indices_begins", block_begins);
            set_i32(request, "la.past_lens", la_past);
            set_i32(request, "la.block_indices", la_blocks);
            set_i32(request, "la.block_indices_begins", la_begins);
            set_i32(request, "la.cache_interval", std::vector<int32_t>(parts.size(), spec ? 1 : 0));
            // last token of each subsequence, or every token
            std::vector<int32_t> rows;
            if (spec) {
                for (int32_t i = 0; i < static_cast<int32_t>(total); ++i)
                    rows.push_back(i);
            } else {
                for (size_t i = 1; i < subseq.size(); ++i)
                    rows.push_back(subseq[i] - 1);
            }
            set_i32(request, "logit_rows", rows);
            request.infer();
            if (spec)
                return;
            for (const auto& [s, len] : parts) {
                slots[s].past += len;
                slots[s].read_slot = 1 - slots[s].read_slot;
            }
        };
        auto argmax_row = [&](const ov::Tensor& logits, size_t row) {
            const auto shape = logits.get_shape();
            const size_t vocab = shape.back();
            const float* r = logits.data<float>() + row * vocab;
            return static_cast<int64_t>(std::distance(r, std::max_element(r, r + vocab)));
        };
        auto is_eos = [&](int64_t t) { return std::find(eos.begin(), eos.end(), t) != eos.end(); };

        // MTP draft step over every active slot: slot s contributes rows
        // [embed(ids[j]) | hid[j]] at positions pos.. (its draft KV holds pos
        // entries); appends the draft token after its last row to s.drafts and
        // leaves that row's hidden state in hid_out[s].
        struct DraftPart {
            int32_t s;
            const std::vector<int64_t>* ids;
            const float* hid;
            int32_t pos;
        };
        std::vector<std::vector<float>> draft_hid(batch, std::vector<float>(mtp_path ? kHidden : 0));
        auto draft_step = [&](const std::vector<DraftPart>& parts) {
            std::vector<int64_t> all_ids;
            for (const auto& p : parts)
                all_ids.insert(all_ids.end(), p.ids->begin(), p.ids->end());
            const size_t total = all_ids.size();
            ov::Tensor input(ov::element::i64, ov::Shape{1, total});
            std::copy(all_ids.begin(), all_ids.end(), input.data<int64_t>());
            embed_request.set_input_tensor(input);
            embed_request.infer();
            const float* emb = embed_request.get_output_tensor().data<float>();
            ov::Tensor x(ov::element::f32, ov::Shape{total, 2 * static_cast<size_t>(kHidden)});
            float* xd = x.data<float>();
            ov::Tensor positions(ov::element::i64, ov::Shape{4, total});
            auto* pos = positions.data<int64_t>();
            std::vector<int32_t> past_lens, subseq{0}, blocks, block_begins{0}, rows;
            size_t tok = 0;
            for (const auto& p : parts) {
                const int32_t n = static_cast<int32_t>(p.ids->size());
                for (int32_t j = 0; j < n; ++j, ++tok) {
                    std::memcpy(xd + tok * 2 * kHidden, emb + tok * kHidden, kHidden * sizeof(float));
                    std::memcpy(xd + tok * 2 * kHidden + kHidden, p.hid + static_cast<size_t>(j) * kHidden,
                                kHidden * sizeof(float));
                    for (size_t section = 0; section < 4; ++section)
                        pos[section * total + tok] = p.pos + j;
                }
                past_lens.push_back(p.pos);
                subseq.push_back(subseq.back() + n);
                rows.push_back(subseq.back() - 1);
                const int32_t active = (p.pos + n + block_size - 1) / block_size;
                for (int32_t b = 0; b < active; ++b)
                    blocks.push_back(p.s * blocks_per_seq + b);
                block_begins.push_back(block_begins.back() + active);
            }
            draft_request.set_tensor("inputs_embeds", x);
            draft_request.set_tensor("position_ids", positions);
            set_i32(draft_request, "past_lens", past_lens);
            set_i32(draft_request, "subsequence_begins", subseq);
            set_i32(draft_request, "block_indices", blocks);
            set_i32(draft_request, "block_indices_begins", block_begins);
            set_i32(draft_request, "logit_rows", rows);
            draft_request.infer();
            const auto logits = draft_request.get_tensor("logits");
            const float* h = draft_request.get_tensor("last_hidden_state").data<float>();
            for (size_t i = 0; i < parts.size(); ++i) {
                slots[parts[i].s].drafts.push_back(argmax_row(logits, i));
                std::memcpy(draft_hid[parts[i].s].data(), h + i * kHidden, kHidden * sizeof(float));
            }
        };
        int64_t mtp_rounds = 0, mtp_accepted = 0;

        std::deque<size_t> pending;
        for (size_t i = 0; i < requests.size(); ++i)
            pending.push_back(i);
        size_t done = 0, steps = 0, decoded_tokens = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto finish = [&](int32_t s) {
            outputs[slots[s].req] = slots[s].generated;
            slots[s].active = false;
            ++done;
            if (verbose)
                std::cerr << "  done " << done << "/" << requests.size() << " req " << slots[s].req << " gen "
                          << slots[s].generated.size() << "\n";
        };

        while (done < requests.size()) {
            // refill free slots: prefill alone, take the first token
            for (int32_t s = 0; s < batch && !pending.empty(); ++s) {
                if (slots[s].active)
                    continue;
                const size_t r = pending.front();
                pending.pop_front();
                slots[s] = Slot{};
                // The paged linear-attention ops always start from the state in
                // block_indices[begin] (past_len == 0 does not imply zero), so a
                // reused slot must have its two state blocks cleared first.
                zero_slot_state(s);
                slots[s].active = true;
                slots[s].req = r;
                feed_embeds(requests[r].prompt);
                run_step({{s, static_cast<int32_t>(requests[r].prompt.size())}});
                const auto logits = request.get_tensor("logits");
                const size_t rows = logits.get_size() / logits.get_shape().back();
                slots[s].next = argmax_row(logits, rows - 1);
                slots[s].generated.push_back(slots[s].next);
                if (is_eos(slots[s].next) || requests[r].max_new <= 1) {
                    finish(s);
                } else if (mtp_path) {
                    // draft prefill rows: target hidden state at j with token j + 1
                    const auto& prompt = requests[r].prompt;
                    slots[s].pend_ids.assign(prompt.begin() + 1, prompt.end());
                    slots[s].pend_ids.push_back(slots[s].next);
                    const auto h = request.get_tensor("last_hidden_state");
                    slots[s].pend_hid.assign(h.data<float>(), h.data<float>() + prompt.size() * kHidden);
                    slots[s].draft_past = 0;
                }
            }
            if (mtp_path) {
                std::vector<int32_t> active;
                for (int32_t s = 0; s < batch; ++s)
                    if (slots[s].active)
                        active.push_back(s);
                if (active.empty())
                    break;
                // draft: pending rows of every slot, then k - 1 single-row steps
                std::vector<DraftPart> dp;
                for (const int32_t s : active) {
                    slots[s].drafts.clear();
                    dp.push_back({s, &slots[s].pend_ids, slots[s].pend_hid.data(), slots[s].draft_past});
                }
                draft_step(dp);
                std::vector<std::vector<int64_t>> one(active.size());
                for (int32_t i = 1; i < mtp_k; ++i) {
                    dp.clear();
                    for (size_t j = 0; j < active.size(); ++j) {
                        const Slot& sl = slots[active[j]];
                        one[j].assign(1, sl.drafts.back());
                        dp.push_back({active[j], &one[j], draft_hid[active[j]].data(),
                                      sl.draft_past + static_cast<int32_t>(sl.pend_ids.size()) - 1 + i});
                    }
                    draft_step(dp);
                }
                // joint verify of [next, d1..dk] per slot
                std::vector<int64_t> tokens;
                std::vector<std::pair<int32_t, int32_t>> parts;
                for (const int32_t s : active) {
                    tokens.push_back(slots[s].next);
                    tokens.insert(tokens.end(), slots[s].drafts.begin(), slots[s].drafts.end());
                    parts.emplace_back(s, mtp_k + 1);
                }
                feed_embeds(tokens);
                run_step(parts, true);
                const auto logits = request.get_tensor("logits");
                const float* hidden = request.get_tensor("last_hidden_state").data<float>();
                ++steps;
                for (size_t j = 0; j < active.size(); ++j) {
                    const int32_t s = active[j];
                    Slot& sl = slots[s];
                    const size_t off = j * static_cast<size_t>(mtp_k + 1);
                    std::vector<int64_t> target(static_cast<size_t>(mtp_k + 1));
                    for (size_t i = 0; i < target.size(); ++i)
                        target[i] = argmax_row(logits, off + i);
                    int32_t a = 0;
                    while (a < mtp_k && sl.drafts[a] == target[a])
                        ++a;
                    ++mtp_rounds;
                    mtp_accepted += a;
                    const auto& req = requests[sl.req];
                    bool stop = false;
                    for (int32_t i = 0; i <= a && !stop; ++i) {
                        const int64_t t = i < a ? sl.drafts[i] : target[a];
                        sl.generated.push_back(t);
                        ++decoded_tokens;
                        stop = is_eos(t) || static_cast<int>(sl.generated.size()) >= req.max_new;
                    }
                    // state after token a: slot order as in run_step(spec)
                    std::vector<int32_t> order{sl.read_slot};
                    for (int32_t o = 0; o < la_slots; ++o)
                        if (o != sl.read_slot)
                            order.push_back(o);
                    sl.read_slot = order[1 + a];
                    sl.pend_ids.assign(sl.drafts.begin(), sl.drafts.begin() + a);
                    sl.pend_ids.push_back(target[a]);
                    sl.pend_hid.assign(hidden + off * kHidden, hidden + (off + a + 1) * kHidden);
                    sl.draft_past = sl.past;
                    sl.past += a + 1;
                    sl.next = target[a];
                    if (stop || sl.past + mtp_k + 1 > max_context_len)
                        finish(s);
                }
                if (verbose && steps % 200 == 0)
                    std::cerr << "  step " << steps << " active " << active.size() << " done " << done
                              << " acceptance " << 100.0 * mtp_accepted / (mtp_rounds * mtp_k) << "%\n";
                continue;
            }
            // joint decode step over the active slots
            std::vector<int32_t> active;
            std::vector<int64_t> tokens;
            std::vector<std::pair<int32_t, int32_t>> parts;
            for (int32_t s = 0; s < batch; ++s) {
                if (!slots[s].active)
                    continue;
                active.push_back(s);
                tokens.push_back(slots[s].next);
                parts.emplace_back(s, 1);
            }
            if (active.empty())
                break;
            feed_embeds(tokens);
            run_step(parts);
            const auto logits = request.get_tensor("logits");
            const size_t rows = logits.get_size() / logits.get_shape().back();
            if (rows != active.size()) {
                std::cerr << "logits rows " << rows << " != active " << active.size() << " shape " << logits.get_shape() << "\n";
                return 1;
            }
            ++steps;
            for (size_t i = 0; i < active.size(); ++i) {
                Slot& sl = slots[active[i]];
                sl.next = argmax_row(logits, i);
                sl.generated.push_back(sl.next);
                ++decoded_tokens;
                const auto& req = requests[sl.req];
                if (is_eos(sl.next) || static_cast<int>(sl.generated.size()) >= req.max_new ||
                    sl.past + 1 >= max_context_len)
                    finish(active[i]);
            }
            if (verbose && steps % 200 == 0)
                std::cerr << "  step " << steps << " active " << active.size() << " done " << done << "\n";
        }
        const double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        std::cerr << "served " << requests.size() << " requests, " << decoded_tokens << " decode tokens in " << steps
                  << " steps, " << secs << " s (" << decoded_tokens / secs << " tok/s aggregate)\n";
        if (mtp_rounds > 0)
            std::cerr << "mtp k=" << mtp_k << ": " << mtp_rounds << " sequence rounds, acceptance "
                      << 100.0 * mtp_accepted / (mtp_rounds * mtp_k) << "%, "
                      << static_cast<double>(mtp_accepted + mtp_rounds) / mtp_rounds << " tokens/round\n";

        std::ofstream out(out_path);
        for (const auto& ids : outputs) {
            for (size_t i = 0; i < ids.size(); ++i)
                out << (i ? "," : "") << ids[i];
            out << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "paged 27b serve failed: " << error.what() << '\n';
        return 1;
    }
}
