/*
 * @Filename: osd-device.cpp
 * @Description: 统一的 OSD 设备实现
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <sys/stat.h>
#include <unistd.h>

#include "osd-device.hpp"
#include "log.hpp"

using namespace fdevice;
namespace sst{
namespace device{
namespace osd{

OsdDevice::OsdDevice()
    : m_height(0), m_width(0) {
    for (int i = 0; i < OSD_LAYER_SIZE; ++i) {
        m_layer_dma[i] = fdevice::DMA_BUFFER_ATTR_S();
        m_layer_has_content[i] = false;
        m_layer_add_failures[i] = 0;
    }
    m_qrangle_out = fdevice::VERTEXS_S();
    m_qrangle_in = fdevice::VERTEXS_S();
}

OsdDevice::~OsdDevice() {
    // Normal demo paths call Release explicitly. Keep an RAII fallback for
    // early returns or partial initialization failures so DMA buffers and LUT
    // memory cannot survive the owner object's lifetime.
    if (m_device_opened || m_osd_enabled || m_pcolor_lut != nullptr) {
        Release();
    }
}

void OsdDevice::Initialize(int width, int height, const char* bitmap_lut_path,
                           int image_dma_size,
                           uint32_t image_layer_mask,
                           uint32_t layer_creation_mask) {
    SigintBlocker sig_blocker;
    if (m_device_opened || m_osd_enabled) {
        Release();
    }
    if (m_pcolor_lut != nullptr) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
    }
    m_width = width;
    m_height = height;

    if (bitmap_lut_path != nullptr && strlen(bitmap_lut_path) > 0) {
        if (LoadLutFile(bitmap_lut_path) != 0) {
            LoadLutFile(m_osd_lut_path.c_str());
        }
    } else {
        LoadLutFile(m_osd_lut_path.c_str());
    }

    if (m_pcolor_lut == nullptr) {
        std::cerr << "[OsdDevice] Error: LUT is null, disabling OSD." << std::endl;
        m_osd_enabled = false;
        return;
    }

    m_osd_handle = osd_open_device();
    if (m_osd_handle == INVALID_HANDLE) {
        std::cerr << "[OsdDevice] Error: osd_open_device failed, disabling OSD."
                  << std::endl;
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        m_osd_enabled = false;
        m_device_opened = false;
        return;
    }
    m_device_opened = true;
    const int init_ret = osd_init_device(
        m_osd_handle, OSD_LAYER_SIZE, (char*)m_pcolor_lut);
    if (init_ret != 0) {
        std::cerr << "[OsdDevice] Error: osd_init_device failed ret="
                  << init_ret << ", disabling OSD." << std::endl;
        osd_close_device(m_osd_handle);
        m_osd_handle = INVALID_HANDLE;
        m_device_opened = false;
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        m_osd_enabled = false;
        return;
    }
    m_osd_enabled = true;

    for (int layer_index = 0; layer_index < OSD_LAYER_SIZE; layer_index++) {
        m_layer_created[layer_index] = false;
        m_layer_has_content[layer_index] = false;
        m_layer_add_failures[layer_index] = 0;
        m_layer_dma[layer_index] = fdevice::DMA_BUFFER_ATTR_S();
        if ((layer_creation_mask & (1u << layer_index)) == 0u) {
            continue;
        }
        // Vector text and multiple tracking boxes can exceed the historical
        // 1 KiB graphic-layer buffer.  This allocation is initialization-only.
        const bool is_image_layer =
            (image_layer_mask & (1u << layer_index)) != 0u;
        // Texture layers used by the menu can need a full-size RLE buffer,
        // while focus tracking only writes compact status sprites. Keep the
        // default for existing modules and allow that mode to request less
        // scarce OCM DMA explicitly.
        int dma_size = 8192;
        if (layer_index == 2) {
            dma_size = std::max(0x1000, image_dma_size);
        } else if (layer_index == 5) {
            // Layer 5 only carries compact icons. A second full-screen pair
            // needlessly fragmented scarce OSD CMA between menu transitions.
            dma_size = std::max(0x1000, std::min(image_dma_size, 0x20000));
        }
        
        const int alloc0_ret = osd_alloc_buffer(
            m_osd_handle, m_layer_dma[layer_index].dma, dma_size);
        if (alloc0_ret != 0 || m_layer_dma[layer_index].dma == nullptr) {
            std::cerr << "[OsdDevice] Warning: first DMA allocation failed for layer "
                      << layer_index << " ret=" << alloc0_ret
                      << ", layer disabled." << std::endl;
            continue;
        }
        usleep(20000);
        const int alloc1_ret = osd_alloc_buffer(
            m_osd_handle, m_layer_dma[layer_index].dma_2, dma_size);
        if (alloc1_ret != 0 || m_layer_dma[layer_index].dma_2 == nullptr) {
            std::cerr << "[OsdDevice] Warning: second DMA allocation failed for layer "
                      << layer_index << " ret=" << alloc1_ret
                      << ", layer disabled." << std::endl;
            osd_delete_buffer(m_osd_handle, m_layer_dma[layer_index].dma);
            m_layer_dma[layer_index].dma = nullptr;
            continue;
        }
        int dma_fd = osd_get_buffer_fd(m_osd_handle, m_layer_dma[layer_index].dma);
        if (dma_fd < 0) {
            std::cerr << "[OsdDevice] Warning: invalid DMA fd for layer "
                      << layer_index << ", layer disabled." << std::endl;
            osd_delete_buffer(m_osd_handle, m_layer_dma[layer_index].dma);
            osd_delete_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2);
            m_layer_dma[layer_index].dma = nullptr;
            m_layer_dma[layer_index].dma_2 = nullptr;
            continue;
        }

        // osd_create_layer() requires every member to be initialized.  In
        // particular, sensor_flag and the inactive encoder descriptor must
        // never contain stack garbage: the OSD driver keeps this descriptor
        // for the complete layer lifetime.
        LAYER_ATTR_S osd_layer = {};
        
        if (is_image_layer) {
            osd_layer.codeTYPE = SS_TYPE_RLE;
            osd_layer.layer_data_RLE.osd_buf.buf_type = BUFFER_TYPE_DMABUF;
            osd_layer.layer_data_RLE.osd_buf.buf.fd_dmabuf = dma_fd;
            osd_layer.layerStart.layer_start_x = 0;
            osd_layer.layerStart.layer_start_y = 0;
            osd_layer.layerSize.layer_width = m_width;
            osd_layer.layerSize.layer_height = m_height;
            osd_layer.layer_rgn = {TYPE_IMAGE, {m_width, m_height}};
        } else {
            osd_layer.codeTYPE = SS_TYPE_QUADRANGLE;
            osd_layer.layer_data_QR.osd_buf.buf_type = BUFFER_TYPE_DMABUF;
            osd_layer.layer_data_QR.osd_buf.buf.fd_dmabuf = dma_fd;
            osd_layer.layerStart.layer_start_x = 0;
            osd_layer.layerStart.layer_start_y = 0;
            osd_layer.layerSize.layer_width = m_width;
            osd_layer.layerSize.layer_height = m_height;
            osd_layer.layer_rgn = {TYPE_GRAPHIC, {m_width, m_height}};
        }

        int cret = osd_create_layer(m_osd_handle, (ssLAYER_HANDLE)layer_index, &osd_layer);
        if (cret == 0) {
            m_layer_created[layer_index] = true;
            const int sret = osd_set_layer_buffer(
                m_osd_handle, (ssLAYER_HANDLE)layer_index,
                m_layer_dma[layer_index]);
            if (sret != 0) {
                std::cerr << "[OsdDevice] Warning: set layer buffer failed for layer "
                          << layer_index << " ret=" << sret << std::endl;
                osd_destroy_layer(m_osd_handle, (ssLAYER_HANDLE)layer_index);
                m_layer_created[layer_index] = false;
                osd_delete_buffer(m_osd_handle, m_layer_dma[layer_index].dma);
                osd_delete_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2);
                m_layer_dma[layer_index].dma = nullptr;
                m_layer_dma[layer_index].dma_2 = nullptr;
            }
        } else {
            std::cerr << "[OsdDevice] Warning: osd_create_layer failed for layer " << layer_index << " ret=" << cret << std::endl;
            m_layer_created[layer_index] = false;
            osd_delete_buffer(m_osd_handle, m_layer_dma[layer_index].dma);
            osd_delete_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2);
            m_layer_dma[layer_index].dma = nullptr;
            m_layer_dma[layer_index].dma_2 = nullptr;
        }
    }

    bool any_created = false;
    for (int i = 0; i < OSD_LAYER_SIZE; ++i) {
        any_created = any_created || m_layer_created[i];
    }
    if (!any_created) {
        std::cerr << "[OsdDevice] Warning: no OSD layers created, disabling OSD." << std::endl;
        Release();
    }
}

void OsdDevice::Release() {
    SigintBlocker sig_blocker;
    // Remove scanout commands before their backing DMA is destroyed. This is
    // essential on A1: destroying a layer that is still referenced by the OSD
    // engine leaves stale scanlines (snow) and can race the next owner.
    for (int i = 0; i < OSD_LAYER_SIZE; ++i) {
        if (m_layer_created[i]) {
            osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)i);
            m_layer_has_content[i] = false;
        }
    }
    usleep(40000);

    for(int i = 0; i < OSD_LAYER_SIZE; i++){
        if (m_layer_created[i]) {
            osd_destroy_layer(m_osd_handle, (ssLAYER_HANDLE)i);
            m_layer_created[i] = false;
        }
        if(m_layer_dma[i].dma != nullptr) {
            osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma);
            m_layer_dma[i].dma = nullptr;
        }
        if(m_layer_dma[i].dma_2 != nullptr) {
            osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma_2);
            m_layer_dma[i].dma_2 = nullptr;
        }
        m_layer_has_content[i] = false;
        m_layer_add_failures[i] = 0;
    }

    if (m_device_opened) {
        osd_close_device(m_osd_handle);
        m_device_opened = false;
    }
    // osd_init_device() receives this pointer. Keep it alive until after the
    // device is closed even if current firmware copies the LUT internally.
    if(m_pcolor_lut != nullptr){
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
    }
    m_osd_enabled = false;
    m_osd_handle = 0; // 重置为无效句柄，防止 dangling handle
    usleep(80000);
}

void OsdDevice::AbortLayerSubmission(int layer_id) {
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE ||
        !m_layer_created[layer_id]) return;
    // An add/flush failure may leave a partial command list even when the
    // first primitive failed. Always clean instead of trusting local state.
    osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
    m_layer_has_content[layer_id] = false;
}

bool OsdDevice::ReserveScanlines(
        const std::array<float, 4>& det,
        std::vector<uint8_t>* scanline_load) const {
    if (scanline_load == nullptr || scanline_load->size() !=
            static_cast<size_t>(m_height) ||
        !std::isfinite(det[1]) || !std::isfinite(det[3])) {
        return false;
    }
    const int y0 = std::max(0, std::min(m_height - 1,
        static_cast<int>(std::floor(std::min(det[1], det[3])))));
    const int y1 = std::max(0, std::min(m_height - 1,
        static_cast<int>(std::ceil(std::max(det[1], det[3])))));
    for (int y = y0; y <= y1; ++y) {
        if ((*scanline_load)[static_cast<size_t>(y)] >= 4u) return false;
    }
    for (int y = y0; y <= y1; ++y) {
        ++(*scanline_load)[static_cast<size_t>(y)];
    }
    return true;
}

int OsdDevice::LoadLutFile(const char* filename){
    LOG_DEBUG("[OsdDevice] Attempting to load LUT file: %s\n", filename);

    struct stat file_stat;
    if (stat(filename, &file_stat) != 0) {
        std::cerr << "[OsdDevice] ERROR: File does not exist or cannot access: " << filename << std::endl;
        return -1;
    }
    if (file_stat.st_size <= 0) {
        std::cerr << "[OsdDevice] ERROR: Invalid file size: " << file_stat.st_size << " bytes" << std::endl;
        return -1;
    }
    if (access(filename, R_OK) != 0) {
        std::cerr << "[OsdDevice] ERROR: No read permission for file: " << filename << std::endl;
        return -1;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);
    if (!file) {
        std::cerr << "[OsdDevice] ERROR: Cannot open file: " << filename << std::endl;
        return -1;
    }

    m_file_size = file.tellg();
    if (m_file_size <= 0) {
        std::cerr << "[OsdDevice] ERROR: Invalid file size from stream." << std::endl;
        file.close();
        return -1;
    }

    m_pcolor_lut = new uint8_t[m_file_size];
    file.seekg(0, std::ios::beg);
    file.read((char*)m_pcolor_lut, m_file_size);

    if (file.gcount() != m_file_size) {
        std::cerr << "[OsdDevice] ERROR: Failed to read complete file." << std::endl;
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        file.close();
        return -1;
    }

    file.close();
    LOG_DEBUG("[OsdDevice] Successfully loaded LUT file, size: %d bytes\n", m_file_size);
    return 0;
}

void OsdDevice::Draw(std::vector<OsdQuadRangle> &quad_rangle){
    if (!m_osd_enabled) return;
    if (quad_rangle.empty()){
        osd_clean_all_layer(m_osd_handle);
        return;
    }

    std::vector<uint8_t> scanline_load(static_cast<size_t>(m_height), 0u);
    for(auto &q : quad_rangle){
        if (!ReserveScanlines(q.box, &scanline_load)) continue;
        if (!GenQrangleBox(q.box, q.border)) continue;
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle(m_osd_handle, &qrangle_attr);
    }
    osd_flush_quad_rangle(m_osd_handle);
}

void OsdDevice::Draw(std::vector<OsdQuadRangle> &quad_rangle, int layer_id){
    if (!m_osd_enabled) return;
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE ||
        !m_layer_created[layer_id]) return;
    if (quad_rangle.empty()){
        ClearLayer(layer_id);
        return;
    }

    // Layer submissions are additive on A1.  Every VISUALIZER call describes
    // the complete state of one layer for the current frame, so replace the
    // previous command list before adding new quadrangles.  Without this,
    // gesture/RPS modes append several primitives at camera rate until the OSD
    // command buffer corrupts the displayed scanlines or driver state.
    const int clear_ret = osd_clean_layer(
        m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (clear_ret != 0) {
        std::cerr << "[OsdDevice] ERROR: clear quadrangle layer failed, layer="
                  << layer_id << " ret=" << clear_ret << std::endl;
        return;
    }
    m_layer_has_content[layer_id] = false;

    std::vector<uint8_t> scanline_load(static_cast<size_t>(m_height), 0u);
    for(auto &q : quad_rangle){
        if (!ReserveScanlines(q.box, &scanline_load)) continue;
        if (!GenQrangleBox(q.box, q.border)) continue;
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        const int ret = osd_add_quad_rangle_layer(
            m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
        if (ret != 0) {
            const uint32_t failures = ++m_layer_add_failures[layer_id];
            if (failures == 1 || failures % 120 == 0) {
                std::cerr << "[OsdDevice] ERROR: add quadrangle failed, layer="
                          << layer_id << " ret=" << ret
                          << " consecutive=" << failures << std::endl;
            }
            AbortLayerSubmission(layer_id);
            return;
        }
        m_layer_has_content[layer_id] = true;
    }
    if (!m_layer_has_content[layer_id]) {
        m_layer_add_failures[layer_id] = 0;
        return;
    }
    const int flush_ret = osd_flush_quad_rangle_layer(
        m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (flush_ret != 0) {
        std::cerr << "[OsdDevice] ERROR: flush quadrangle failed, layer="
                  << layer_id << " ret=" << flush_ret << std::endl;
        AbortLayerSubmission(layer_id);
        return;
    }
    m_layer_add_failures[layer_id] = 0;
}

void OsdDevice::ClearLayer(int layer_id) {
    if (!m_osd_enabled || layer_id < 0 || layer_id >= OSD_LAYER_SIZE ||
        !m_layer_created[layer_id] || !m_layer_has_content[layer_id]) return;
    const int ret = osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (ret != 0) {
        std::cerr << "[OsdDevice] ERROR: clear layer failed, layer="
                  << layer_id << " ret=" << ret << std::endl;
    } else {
        m_layer_has_content[layer_id] = false;
    }
}

void OsdDevice::Draw(std::vector<std::array<float, 4>>& boxes, int border, int layer_id, fdevice::QUADRANGLETYPE type, fdevice::ALPHATYPE alpha, int color){
    if (!m_osd_enabled) return;
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE ||
        !m_layer_created[layer_id]) return;
    if (boxes.empty()){
        ClearLayer(layer_id);
        return;
    }

    const int clear_ret = osd_clean_layer(
        m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (clear_ret != 0) {
        std::cerr << "[OsdDevice] ERROR: clear quadrangle layer failed, layer="
                  << layer_id << " ret=" << clear_ret << std::endl;
        return;
    }
    m_layer_has_content[layer_id] = false;

    std::vector<uint8_t> scanline_load(static_cast<size_t>(m_height), 0u);
    for (auto &box : boxes){
        if (!ReserveScanlines(box, &scanline_load)) continue;
        if (!GenQrangleBox(box, border)) continue;
        COVER_ATTR_S qrangle_attr = {color, type, alpha, m_qrangle_out, m_qrangle_in};
        const int ret = osd_add_quad_rangle_layer(
            m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
        if (ret != 0) {
            const uint32_t failures = ++m_layer_add_failures[layer_id];
            if (failures == 1 || failures % 120 == 0) {
                std::cerr << "[OsdDevice] ERROR: add quadrangle failed, layer="
                          << layer_id << " ret=" << ret
                          << " consecutive=" << failures << std::endl;
            }
            AbortLayerSubmission(layer_id);
            return;
        }
        m_layer_has_content[layer_id] = true;
    }
    if (!m_layer_has_content[layer_id]) {
        m_layer_add_failures[layer_id] = 0;
        return;
    }
    const int flush_ret = osd_flush_quad_rangle_layer(
        m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (flush_ret != 0) {
        std::cerr << "[OsdDevice] ERROR: flush quadrangle failed, layer="
                  << layer_id << " ret=" << flush_ret << std::endl;
        AbortLayerSubmission(layer_id);
        return;
    }
    m_layer_add_failures[layer_id] = 0;
}

void OsdDevice::DrawTexture(const char* bitmap_path, const char* lut_path, int layer_id, int pos_x, int pos_y, fdevice::ALPHATYPE alpha) {
    (void)lut_path;
    if (!m_osd_enabled) return;
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE || !m_layer_created[layer_id]) {
        std::cerr << "[OsdDevice] ERROR: Layer " << layer_id << " not created, skipping DrawTexture." << std::endl;
        return;
    }
    if (bitmap_path == nullptr || access(bitmap_path, F_OK) != 0) {
        std::cerr << "[OsdDevice] ERROR: Bitmap file not found: " << (bitmap_path ? bitmap_path : "null") << std::endl;
        return;
    }

    // Keep every reserved field deterministic; the driver may retain this
    // descriptor until the next layer update.
    fdevice::BITMAP_INFO_S bm_info = {};
    bm_info.pSSbmpFile = bitmap_path;
    bm_info.alpha = alpha;
    bm_info.position.x = pos_x;
    bm_info.position.y = pos_y;

    const int clear_ret = osd_clean_layer(
        m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (clear_ret != 0) {
        std::cerr << "[OsdDevice] ERROR: clear texture layer failed, layer="
                  << layer_id << " ret=" << clear_ret << std::endl;
        return;
    }
    m_layer_has_content[layer_id] = false;

    int ret = osd_add_texture_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &bm_info);
    if (ret != 0) {
        std::cerr << "[OsdDevice] ERROR: osd_add_texture_layer failed! layer_id=" << layer_id << std::endl;
        AbortLayerSubmission(layer_id);
        return;
    }
    m_layer_has_content[layer_id] = true;
    const int flush_ret = osd_flush_texture_layer(
        m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (flush_ret != 0) {
        std::cerr << "[OsdDevice] ERROR: flush texture failed, layer="
                  << layer_id << " ret=" << flush_ret << std::endl;
        AbortLayerSubmission(layer_id);
    }
}

bool OsdDevice::GenQrangleBox(const std::array<float, 4>& det, int border){
    if (m_width <= 1 || m_height <= 1 ||
        !std::isfinite(det[0]) || !std::isfinite(det[1]) ||
        !std::isfinite(det[2]) || !std::isfinite(det[3])) {
        return false;
    }

    const int max_x = m_width - 1;
    const int max_y = m_height - 1;
    const int x1 = std::max(0, std::min(max_x,
        static_cast<int>(std::floor(std::min(det[0], det[2])))));
    const int y1 = std::max(0, std::min(max_y,
        static_cast<int>(std::floor(std::min(det[1], det[3])))));
    const int x2 = std::max(0, std::min(max_x,
        static_cast<int>(std::ceil(std::max(det[0], det[2])))));
    const int y2 = std::max(0, std::min(max_y,
        static_cast<int>(std::ceil(std::max(det[1], det[3])))));
    border = std::max(0, border);
    if (x2 - x1 < std::max(1, border * 2 + 1) ||
        y2 - y1 < std::max(1, border * 2 + 1)) {
        return false;
    }

    std::array<int, 16> box;

    box[0] = std::min(max_x, x1 + border);
    box[1] = std::min(max_y, y1 + border);
    box[2] = std::min(max_x, x1 + border);
    box[3] = std::max(0, y2 - border);
    box[4] = std::max(0, x2 - border);
    box[5] = std::max(0, y2 - border);
    box[6] = std::max(0, x2 - border);
    box[7] = std::min(max_y, y1 + border);

    box[8] = std::max(0, x1 - border);
    box[9] = std::max(0, y1 - border);
    box[10] = std::max(0, x1 - border);
    box[11] = std::min(max_y, y2 + border);
    box[12] = std::min(max_x, x2 + border);
    box[13] = std::min(max_y, y2 + border);
    box[14] = std::min(max_x, x2 + border);
    box[15] = std::max(0, y1 - border);

    m_qrangle_in.points[0]={box[0], box[1]};
    m_qrangle_in.points[1]={box[2], box[3]};
    m_qrangle_in.points[2]={box[4], box[5]};
    m_qrangle_in.points[3]={box[6], box[7]};
    m_qrangle_out.points[0] = {box[8], box[9]};
    m_qrangle_out.points[1] = {box[10], box[11]};
    m_qrangle_out.points[2] = {box[12], box[13]};
    m_qrangle_out.points[3] = {box[14], box[15]};
    return true;
}

} // namespace osd
} // namespace device
} // namespace sst
