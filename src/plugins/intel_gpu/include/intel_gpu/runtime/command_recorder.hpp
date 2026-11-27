// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "command_list.hpp"
#include "openvino/core/except.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace cldnn {
// SCRATCH (OV_GPU_REPLAY diagnostics): primitive being prepared/executed, and recording abort sites.
inline thread_local const std::string* g_rec_cur_prim = nullptr;
inline std::map<std::string, size_t>& rec_abort_sites() {
    static std::map<std::string, size_t> m;
    return m;
}
// SCRATCH: > 0 while the host writes data that depends only on shapes (shape_info); such a lock keeps a recording.
inline thread_local int g_rec_benign_lock = 0;
struct rec_benign_lock_scope {
    rec_benign_lock_scope() { ++g_rec_benign_lock; }
    ~rec_benign_lock_scope() { --g_rec_benign_lock; }
};
inline void note_rec_abort(const char* kind) {
    rec_abort_sites()[std::string(kind) + " @ " + (g_rec_cur_prim ? *g_rec_cur_prim : std::string("?"))]++;
}
/// @brief Interface for recording commands on associated stream.
/// @note command_recorder interface is not thread safe and must be used from the same thread as the associated stream
class command_recorder {
public:
    using ptr = std::shared_ptr<command_recorder>;
    command_recorder() = default;
    virtual ~command_recorder() = default;
    // Delete copy ctor to prevent sharing active command list
    command_recorder(const command_recorder& other) = delete;

    /// @brief Create command list for recording executed commands.
    /// @return Command list object.
    virtual command_list::ptr create_command_list() const = 0;

    /// @brief Start recording operations executed on the associated stream.
    /// Executed commands are not submitted to the device during recording.
    /// @param cmd_list Open command list to record executed commands.
    void start_recording(command_list::ptr cmd_list) {
        OPENVINO_ASSERT(cmd_list->get_status() == command_list_status::open, "Can't start recording command list that is not open");
        OPENVINO_ASSERT(_active_cmd_list == nullptr, "Can't start recording while another recording is in progress");
        _active_cmd_list = cmd_list;
    }
    /// @brief Get command list that is currently being recorded.
    /// @return Command list that is currently being recorded or nullptr.
    command_list::ptr get_active_command_list() const {
        return _active_cmd_list;
    }

    /// @brief Stop recording and submit all recorded commands to the device.
    /// @return Command list with recorded commands or nullptr if stream was not recording.
    command_list::ptr stop_recording() {
        command_list::ptr ret = nullptr;
        _active_cmd_list.swap(ret);
        if (ret != nullptr) {
            ret->close();
            ret->enqueue();
        }
        return ret;
    }

protected:
    command_list::ptr _active_cmd_list;
};

}  // namespace cldnn
