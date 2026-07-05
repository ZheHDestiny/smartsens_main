/*
 * @Filename: osd-device.cpp
 * @Description: 统一的 OSD 设备实现
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cerrno>
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
    }
    m_qrangle_out = fdevice::VERTEXS_S();
    m_qrangle_in = fdevice::VERTEXS_S();
}

OsdDevice::~OsdDevice() {
}

void OsdDevice::Initialize(int width, int height, const char* bitmap_lut_path) {
    SigintBlocker sig_blocker;
    if (m_osd_enabled) {
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
    osd_init_device(m_osd_handle, OSD_LAYER_SIZE, (char*)m_pcolor_lut);
    m_osd_enabled = true;

    for (int layer_index = 0; layer_index < OSD_LAYER_SIZE; layer_index++) {
        int dma_size = (layer_index == 2) ? 0x100000 : 1024;
        
        osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma, dma_size);
        usleep(250000); // 等待 DMA 分配稳定
        osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2, dma_size);
        int dma_fd = osd_get_buffer_fd(m_osd_handle, m_layer_dma[layer_index].dma);

        LAYER_ATTR_S osd_layer;
        
        if (layer_index == 2) {
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
            osd_set_layer_buffer(m_osd_handle, (ssLAYER_HANDLE)layer_index, m_layer_dma[layer_index]);
        } else {
            std::cerr << "[OsdDevice] Warning: osd_create_layer failed for layer " << layer_index << " ret=" << cret << std::endl;
            m_layer_created[layer_index] = false;
        }
    }

    bool any_created = false;
    for (int i = 0; i < OSD_LAYER_SIZE; ++i) {
        any_created = any_created || m_layer_created[i];
    }
    if (!any_created) {
        std::cerr << "[OsdDevice] Warning: no OSD layers created, disabling OSD." << std::endl;
        m_osd_enabled = false;
    }
}

void OsdDevice::Release() {
    SigintBlocker sig_blocker;
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
    }

    if(m_pcolor_lut != nullptr){
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
    }

    if (m_osd_enabled) {
        osd_close_device(m_osd_handle);
        m_osd_enabled = false;
    }
    m_osd_handle = 0; // 重置为无效句柄，防止 dangling handle
    usleep(300000); // 等待驱动完成资源清理
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

    for(auto &q : quad_rangle){
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle(m_osd_handle, &qrangle_attr);
    }
    osd_flush_quad_rangle(m_osd_handle);
}

void OsdDevice::Draw(std::vector<OsdQuadRangle> &quad_rangle, int layer_id){
    if (!m_osd_enabled) return;
    if (quad_rangle.empty()){
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        return;
    }

    for(auto &q : quad_rangle){
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
    }
    osd_flush_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
}

void OsdDevice::Draw(std::vector<std::array<float, 4>>& boxes, int border, int layer_id, fdevice::QUADRANGLETYPE type, fdevice::ALPHATYPE alpha, int color){
    if (!m_osd_enabled) return;
    if (boxes.empty()){
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        return;
    }

    for (auto &box : boxes){
        GenQrangleBox(box, border);
        COVER_ATTR_S qrangle_attr = {color, type, alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
    }
    osd_flush_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
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

    fdevice::BITMAP_INFO_S bm_info;
    bm_info.pSSbmpFile = bitmap_path;
    bm_info.alpha = alpha;
    bm_info.position.x = pos_x;
    bm_info.position.y = pos_y;

    int ret = osd_add_texture_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &bm_info);
    if (ret != 0) {
        std::cerr << "[OsdDevice] ERROR: osd_add_texture_layer failed! layer_id=" << layer_id << std::endl;
        return;
    }
    osd_flush_texture_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
}

void OsdDevice::GenQrangleBox(std::array<float, 4>& det, int border){
    std::array<int, 16> box;

    box[0] = std::min(m_width, std::max(0, int(det[0]+border)));
    box[1] = std::min(m_height, std::max(0, int(det[1]+border)));
    box[2] = std::min(m_width, std::max(0, int(det[0]+border)));
    box[3] = std::min(m_height, std::max(0, int(det[3]-border)));
    box[4] = std::min(m_width, std::max(0, int(det[2]-border)));
    box[5] = std::min(m_height, std::max(0, int(det[3]-border)));
    box[6] = std::min(m_width, std::max(0, int(det[2]-border)));
    box[7] = std::min(m_height, std::max(0, int(det[1]+border)));

    box[8] = std::min(m_width, std::max(0, int(det[0]-border)));
    box[9] = std::min(m_height, std::max(0, int(det[1]-border)));
    box[10] = std::min(m_width, std::max(0, int(det[0]-border)));
    box[11] = std::min(m_height, std::max(0, int(det[3]+border)));
    box[12] = std::min(m_width, std::max(0, int(det[2]+border)));
    box[13] = std::min(m_height, std::max(0, int(det[3]+border)));
    box[14] = std::min(m_width, std::max(0, int(det[2]+border)));
    box[15] = std::min(m_height, std::max(0, int(det[1]-border)));

    m_qrangle_in.points[0]={box[0], box[1]};
    m_qrangle_in.points[1]={box[2], box[3]};
    m_qrangle_in.points[2]={box[4], box[5]};
    m_qrangle_in.points[3]={box[6], box[7]};
    m_qrangle_out.points[0] = {box[8], box[9]};
    m_qrangle_out.points[1] = {box[10], box[11]};
    m_qrangle_out.points[2] = {box[12], box[13]};
    m_qrangle_out.points[3] = {box[14], box[15]};
}

} // namespace osd
} // namespace device
} // namespace sst
