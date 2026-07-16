/*
 * @Filename: osd-device.hpp
 * @Description: 统一的 OSD 设备接口
 */
#ifndef SST_OSD_DEVICE_HPP_
#define SST_OSD_DEVICE_HPP_

#include <vector>
#include <string>
#include <array>

#ifdef TEST_PC_BUILD
#include <cstdint>
typedef int handle_t;
typedef int ssLAYER_HANDLE;
#define SS_TYPE_QUADRANGLE 0
#define SS_TYPE_RLE 1
#define TYPE_GRAPHIC 0
#define TYPE_IMAGE 1
namespace fdevice {
    typedef enum { TYPE_HOLLOW = 0, TYPE_SOLID = 1 } QUADRANGLETYPE;
    typedef enum { TYPE_ALPHA25 = 0, TYPE_ALPHA50, TYPE_ALPHA75, TYPE_ALPHA100 } ALPHATYPE;
    struct POINT_S { int x; int y; };
    struct VERTEXS_S { POINT_S points[4]; };
    struct DMA_BUFFER_ATTR_S { void* dma = nullptr; void* dma_2 = nullptr; };
    struct OSD_BUF_S { int buf_type; union { int fd_dmabuf; } buf; };
    struct LAYER_DATA_QR_S { OSD_BUF_S osd_buf; };
    struct LAYER_DATA_RLE_S { OSD_BUF_S osd_buf; };
    struct LAYER_START_S { int layer_start_x; int layer_start_y; };
    struct LAYER_SIZE_S { int layer_width; int layer_height; };
    struct LAYER_RGN_S { int type; LAYER_SIZE_S size; };
    struct LAYER_ATTR_S {
        int codeTYPE;
        LAYER_DATA_QR_S layer_data_QR;
        LAYER_DATA_RLE_S layer_data_RLE;
        LAYER_START_S layerStart;
        LAYER_SIZE_S layerSize;
        LAYER_RGN_S layer_rgn;
    };
    struct COVER_ATTR_S {
        int colorIdx; QUADRANGLETYPE eSolid; ALPHATYPE alpha;
        VERTEXS_S vertex_out; VERTEXS_S vertex_in;
    };
    struct POSITION_S { int x; int y; };
    struct BITMAP_INFO_S { const char* pSSbmpFile; ALPHATYPE alpha; POSITION_S position; };
}
using namespace fdevice;
inline handle_t osd_open_device() { return 0; }
inline void osd_init_device(handle_t, int, char*) {}
inline void osd_alloc_buffer(handle_t, void*&, int) {}
inline int osd_get_buffer_fd(handle_t, void*) { return 0; }
inline int osd_create_layer(handle_t, ssLAYER_HANDLE, LAYER_ATTR_S*) { return 0; }
inline int osd_set_layer_buffer(handle_t, ssLAYER_HANDLE, DMA_BUFFER_ATTR_S) { return 0; }
inline void osd_destroy_layer(handle_t, ssLAYER_HANDLE) {}
inline void osd_delete_buffer(handle_t, void*) {}
inline void osd_close_device(handle_t) {}
inline int osd_add_quad_rangle(handle_t, COVER_ATTR_S*) { return 0; }
inline int osd_flush_quad_rangle(handle_t) { return 0; }
inline int osd_add_quad_rangle_layer(handle_t, ssLAYER_HANDLE, COVER_ATTR_S*) { return 0; }
inline int osd_flush_quad_rangle_layer(handle_t, ssLAYER_HANDLE) { return 0; }
inline int osd_clean_layer(handle_t, ssLAYER_HANDLE) { return 0; }
inline int osd_clean_all_layer(handle_t) { return 0; }
inline int osd_add_texture_layer(handle_t, ssLAYER_HANDLE, BITMAP_INFO_S*) { return 0; }
inline int osd_flush_texture_layer(handle_t, ssLAYER_HANDLE) { return 0; }
#else
#include "osd_lib_api.h"
#endif

#include "common.hpp"

#define BUFFER_TYPE_DMABUF  0x1
#define OSD_LAYER_SIZE 6

namespace sst {
namespace device {
namespace osd {

typedef struct {
    std::array<float, 4> box;
    int border;
    int layer_id;
    fdevice::QUADRANGLETYPE type;
    fdevice::ALPHATYPE alpha;
    int color;
} OsdQuadRangle;

class OsdDevice {
public:
    OsdDevice();
    ~OsdDevice();

    void Initialize(int width, int height, const char* bitmap_lut_path = nullptr);
    void Release();
    bool IsEnabled() const { return m_osd_enabled; }

    void Draw(std::vector<OsdQuadRangle> &quad_rangle);
    void Draw(std::vector<std::array<float, 4>>& boxes, int border, int layer_id, fdevice::QUADRANGLETYPE type, fdevice::ALPHATYPE alpha, int color);
    void Draw(std::vector<OsdQuadRangle> &quad_rangle, int layer_id);
    void ClearLayer(int layer_id);
    void DrawTexture(const char* bitmap_path, const char* lut_path, int layer_id, int pos_x = 0, int pos_y = 0, fdevice::ALPHATYPE alpha = fdevice::TYPE_ALPHA100);

private:
    int LoadLutFile(const char* filename);
    void GenQrangleBox(std::array<float, 4>& det, int border);

private:
    handle_t m_osd_handle = 0;
    std::string m_osd_lut_path = "./app_assets/colorLUT.sscl";
    uint8_t *m_pcolor_lut = nullptr;
    int m_file_size = 0;
    int m_height, m_width;

    fdevice::DMA_BUFFER_ATTR_S m_layer_dma[OSD_LAYER_SIZE];
    bool m_layer_created[OSD_LAYER_SIZE] = {false};
    bool m_osd_enabled = false;
    fdevice::VERTEXS_S m_qrangle_out, m_qrangle_in;
};

} // namespace osd
} // namespace device
} // namespace sst

#endif // SST_OSD_DEVICE_HPP_
