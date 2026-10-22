// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

/*
 * Gemma 3 attention input, as exported from HF: q/k are viewed as heads, transposed, RMS-normalized
 * per head (q_norm / k_norm) and rotated (rotate-half RoPE over the whole head). The global and the
 * sliding-window rotary embeddings are two MatMul -> Sin/Cos branches with different frequencies
 * that share one Convert of the position ids:
 *
 *                        position_ids -> Unsqueeze -> Convert(f32)
 *                                                 /            \
 *          Broadcast(inv_freq_global) -> MatMul               MatMul <- Broadcast(inv_freq_local)
 *                                          |                    |
 *                          Transpose -> Concat -> Sin/Cos      ... -> Sin/Cos
 *                                          |                    |
 *   q -> Reshape -> Transpose -> RMS -> RoPE(q)       k -> ... -> RoPE(k)
 *
 * With RMS + RoPE folded into one RoPE (FuseRMSRoPE), the rotary MatMuls must still get consistent
 * input precisions from IncreasePositionIdsPrecision (CVS-197848).
 */
constexpr int64_t kHeadSize = 64;
constexpr int64_t kQHeads = 2;
constexpr int64_t kKVHeads = 1;

class RMSRoPESharedPositionIds : public testing::WithParamInterface<std::vector<InputShape>>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<InputShape>>& obj) {
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : obj.param) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : obj.param) {
            for (const auto& s : shape.second) {
                result << ov::test::utils::vec2str(s) << "_";
            }
        }
        return result.str();
    }

protected:
    static ov::Output<ov::Node> rms(const ov::Output<ov::Node>& x, size_t seed) {
        auto power = std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(ov::element::f32, {}, {2.f}));
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), true);
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, {}, {1e-6f}));
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);
        auto div = std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, {}, {1.f}), sqrt);
        auto norm = std::make_shared<ov::op::v1::Multiply>(x, div);
        // Gemma's (1 + weight): a non-trivial gamma, otherwise RMS would be fused without one.
        auto gamma = ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                             ov::Shape{static_cast<size_t>(kHeadSize)},
                                                             ov::test::utils::InputGenerateData(0, 2, 1000, static_cast<int>(seed)));
        return std::make_shared<ov::op::v1::Multiply>(norm, std::make_shared<ov::op::v0::Constant>(gamma));
    }

    // [B, 1, S, D] cos and sin of position * inv_freq, emb = concat(freqs, freqs).
    static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> cos_sin(const ov::Output<ov::Node>& pos_f32, float base) {
        std::vector<float> inv_freq(kHeadSize / 2);
        for (size_t i = 0; i < inv_freq.size(); ++i) {
            inv_freq[i] = 1.f / std::pow(base, 2.f * static_cast<float>(i) / static_cast<float>(kHeadSize));
        }
        auto inv = ov::op::v0::Constant::create(ov::element::f32, {1, static_cast<size_t>(kHeadSize / 2), 1}, inv_freq);
        auto batch = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(pos_f32),
                                                          ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                          ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto target = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{batch, ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{kHeadSize / 2, 1})},
            0);
        auto inv_b = std::make_shared<ov::op::v3::Broadcast>(inv, target);
        auto freqs = std::make_shared<ov::op::v0::MatMul>(inv_b, pos_f32);  // [B, D/2, S]
        auto freqs_t = std::make_shared<ov::op::v1::Transpose>(freqs, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 2, 1}));
        auto emb = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{freqs_t, freqs_t}, -1);  // [B, S, D]
        auto axis1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        return {std::make_shared<ov::op::v0::Unsqueeze>(std::make_shared<ov::op::v0::Cos>(emb), axis1),
                std::make_shared<ov::op::v0::Unsqueeze>(std::make_shared<ov::op::v0::Sin>(emb), axis1)};
    }

    static ov::Output<ov::Node> slice_last(const ov::Output<ov::Node>& x, int64_t start, int64_t stop) {
        return std::make_shared<ov::op::v8::Slice>(x,
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {start}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {stop}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    }

    // [B, S, heads * D] -> heads -> RMS -> rotate-half RoPE, [B, heads, S, D]
    static ov::Output<ov::Node> head_path(const ov::Output<ov::Node>& x,
                                          int64_t heads,
                                          const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& cs,
                                          size_t seed) {
        auto view =
            std::make_shared<ov::op::v1::Reshape>(x, ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, heads, kHeadSize}), true);
        auto t = std::make_shared<ov::op::v1::Transpose>(view, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        auto n = rms(t, seed);
        auto x1 = slice_last(n, 0, kHeadSize / 2);
        auto x2 = slice_last(n, kHeadSize / 2, std::numeric_limits<int32_t>::max());
        auto x2neg = std::make_shared<ov::op::v1::Multiply>(x2, ov::op::v0::Constant::create(ov::element::f32, {}, {-1.f}));
        auto rotated = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2neg, x1}, -1);
        auto mul_cos = std::make_shared<ov::op::v1::Multiply>(n, cs.first);
        auto mul_sin = std::make_shared<ov::op::v1::Multiply>(rotated, cs.second);
        return std::make_shared<ov::op::v1::Add>(mul_cos, mul_sin);
    }

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        for (size_t i = 0; i < 2; ++i) {
            inputs[params[i]] = ov::test::utils::create_and_fill_tensor(ov::element::f32, target_shapes[i], ov::test::utils::InputGenerateData(-1, 2, 1000, 1));
        }
        ov::Tensor pos(ov::element::i64, target_shapes[2]);
        const size_t seq = target_shapes[2][1];
        for (size_t i = 0; i < pos.get_size(); ++i) {
            pos.data<int64_t>()[i] = static_cast<int64_t>(i % seq + 3);
        }
        inputs[params[2]] = pos;
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes(GetParam());
        configuration[ov::hint::inference_precision.name()] = ov::element::f16;
        abs_threshold = 0.03f;

        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1]);
        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[2]);
        position_ids->set_friendly_name("position_ids");
        position_ids->output(0).set_names({"position_ids"});

        auto pos = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
        auto pos_f32 = std::make_shared<ov::op::v0::Convert>(pos, ov::element::f32);  // shared by both rotary branches
        auto global = cos_sin(pos_f32, 1000000.f);
        auto local = cos_sin(pos_f32, 10000.f);

        auto q_out = head_path(q, kQHeads, global, 1);
        auto k_out = head_path(k, kKVHeads, local, 2);
        function = std::make_shared<ov::Model>(ov::OutputVector{q_out, k_out}, ov::ParameterVector{q, k, position_ids}, "RMSRoPESharedPositionIds");
    }
};

TEST_P(RMSRoPESharedPositionIds, Inference) {
    run();
    // Both per-head RMSNorms are folded into their RoPE, which is the configuration this test covers.
    ov::test::CheckNumberOfNodesWithType(compiledModel, "RoPE", 2);
    ov::test::CheckNumberOfNodesWithType(compiledModel, "RMS", 0);
}

const std::vector<std::vector<InputShape>> input_shapes = {
    {{{-1, -1, kQHeads* kHeadSize}, {{1, 9, kQHeads* kHeadSize}, {2, 1, kQHeads* kHeadSize}}},
     {{-1, -1, kKVHeads* kHeadSize}, {{1, 9, kKVHeads* kHeadSize}, {2, 1, kKVHeads* kHeadSize}}},
     {{-1, -1}, {{1, 9}, {2, 1}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RMSRoPESharedPositionIds,
                         RMSRoPESharedPositionIds,
                         ::testing::ValuesIn(input_shapes),
                         RMSRoPESharedPositionIds::getTestCaseName);
}  // namespace
