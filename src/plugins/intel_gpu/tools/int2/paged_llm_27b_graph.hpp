// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Graph edits shared by the paged 27B tools (target + MTP draft), applied
// in-process after SDPAToPagedAttention.

#pragma once

#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace paged27b {

// Target: input of the lm_head's Hadamard rotation = post-final-norm hidden
// state (sign-flipped by the folded rotation, see bonsai2_mtp_to_ir.py).
constexpr const char* kFinalNorm = "language_model.norm/aten::mul/Multiply_1";
// MTP draft: output of its one decoder layer (input of mtp.norm).
constexpr const char* kDraftLayerOut = "language_model.layers.3/aten::add/Add_1";

inline std::shared_ptr<ov::Node> find_by_suffix(const std::shared_ptr<ov::Model>& model, const std::string& suffix) {
    std::shared_ptr<ov::Node> found;
    for (const auto& node : model->get_ordered_ops()) {
        const auto& n = node->get_friendly_name();
        if (n.size() >= suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
            found = node;
    }
    if (!found)
        throw std::runtime_error("node *" + suffix + " not found");
    return found;
}

// Gather the rows given by a new i32 input `logit_rows` from `node` for all
// its current consumers. The token axis is the dynamic one ([tokens, 1, hidden]
// on the paged path).
inline void add_row_gather(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<ov::Node>& node) {
    const auto& pshape = node->get_output_partial_shape(0);
    int64_t token_axis = -1;
    for (int64_t i = 0; i < pshape.rank().get_length(); ++i)
        if (pshape[i].is_dynamic()) {
            token_axis = i;
            break;
        }
    if (token_axis < 0)
        throw std::runtime_error("no dynamic token axis on " + node->get_friendly_name() + " " + pshape.to_string());
    const auto consumers = node->output(0).get_target_inputs();  // before the Gather joins them
    auto rows = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    rows->set_friendly_name("logit_rows");
    rows->output(0).set_names({"logit_rows"});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {token_axis});
    auto gather = std::make_shared<ov::op::v8::Gather>(node->output(0), rows, axis);
    for (auto target : consumers)
        target.replace_source_output(gather->output(0));
    model->add_parameters({rows});
    model->validate_nodes_and_infer_types();
}

inline void add_output(const std::shared_ptr<ov::Model>& model, const ov::Output<ov::Node>& out, const std::string& name) {
    out.get_tensor().add_names({name});
    model->add_results({std::make_shared<ov::op::v0::Result>(out)});
}

}  // namespace paged27b
