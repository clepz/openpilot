#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>

#include <czmq.h>

#include <pthread.h>

#include <OMX_Component.h>
#include <OMX_IndexExt.h>
#include <OMX_VideoExt.h>
#include <OMX_QCOMExtns.h>

#include <libyuv.h>

#include <android/log.h>

#include <msm_media_info.h>

#include "common/mutex.h"
#include "common/swaglog.h"

#include "encoder.h"

#define ALOG(...) __android_log_print(ANDROID_LOG_VERBOSE, "omxapp", ##__VA_ARGS__)

// encoder: lossey codec using hardware hevc

static void wait_for_state(EncoderState *s, OMX_STATETYPE state) {
  pthread_mutex_lock(&s->state_lock);
  while (s->state != state) {
    pthread_cond_wait(&s->state_cv, &s->state_lock);
  }
  pthread_mutex_unlock(&s->state_lock);
}

static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data) {
  EncoderState *s = app_data;

  switch (event) {
  case OMX_EventCmdComplete:
    assert(data1 == OMX_CommandStateSet);
    LOG("set state event 0x%x", data2);
    pthread_mutex_lock(&s->state_lock);
    s->state = data2;
    pthread_cond_broadcast(&s->state_cv);
    pthread_mutex_unlock(&s->state_lock);
    break;
  case OMX_EventError:
    LOGE("OMX error 0x%08x", data1);
    // assert(false);
    break;
  default:
    LOGE("unhandled event %d", event);
    assert(false);
    break;
  }

  pthread_mutex_unlock(&s->state_lock);
  
  return OMX_ErrorNone;
}

static OMX_ERRORTYPE empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                       OMX_BUFFERHEADERTYPE *buffer) {
  EncoderState *s = app_data;

  // printf("empty_buffer_done\n");

  queue_push(&s->free_in, (void*)buffer);

  return OMX_ErrorNone;
}


static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                      OMX_BUFFERHEADERTYPE *buffer) {
  EncoderState *s = app_data;

  // printf("fill_buffer_done\n");

  queue_push(&s->done_out, (void*)buffer);

  return OMX_ErrorNone;
}

static OMX_CALLBACKTYPE omx_callbacks = {
  .EventHandler = event_handler,
  .EmptyBufferDone = empty_buffer_done,
  .FillBufferDone = fill_buffer_done,
};

#define PORT_INDEX_IN 0
#define PORT_INDEX_OUT 1

static const char* omx_color_fomat_name(uint32_t format) {
  switch (format) {
  case OMX_COLOR_FormatUnused: return "OMX_COLOR_FormatUnused";
  case OMX_COLOR_FormatMonochrome: return "OMX_COLOR_FormatMonochrome";
  case OMX_COLOR_Format8bitRGB332: return "OMX_COLOR_Format8bitRGB332";
  case OMX_COLOR_Format12bitRGB444: return "OMX_COLOR_Format12bitRGB444";
  case OMX_COLOR_Format16bitARGB4444: return "OMX_COLOR_Format16bitARGB4444";
  case OMX_COLOR_Format16bitARGB1555: return "OMX_COLOR_Format16bitARGB1555";
  case OMX_COLOR_Format16bitRGB565: return "OMX_COLOR_Format16bitRGB565";
  case OMX_COLOR_Format16bitBGR565: return "OMX_COLOR_Format16bitBGR565";
  case OMX_COLOR_Format18bitRGB666: return "OMX_COLOR_Format18bitRGB666";
  case OMX_COLOR_Format18bitARGB1665: return "OMX_COLOR_Format18bitARGB1665";
  case OMX_COLOR_Format19bitARGB1666: return "OMX_COLOR_Format19bitARGB1666"; 
  case OMX_COLOR_Format24bitRGB888: return "OMX_COLOR_Format24bitRGB888";
  case OMX_COLOR_Format24bitBGR888: return "OMX_COLOR_Format24bitBGR888";
  case OMX_COLOR_Format24bitARGB1887: return "OMX_COLOR_Format24bitARGB1887";
  case OMX_COLOR_Format25bitARGB1888: return "OMX_COLOR_Format25bitARGB1888";
  case OMX_COLOR_Format32bitBGRA8888: return "OMX_COLOR_Format32bitBGRA8888";
  case OMX_COLOR_Format32bitARGB8888: return "OMX_COLOR_Format32bitARGB8888";
  case OMX_COLOR_FormatYUV411Planar: return "OMX_COLOR_FormatYUV411Planar";
  case OMX_COLOR_FormatYUV411PackedPlanar: return "OMX_COLOR_FormatYUV411PackedPlanar";
  case OMX_COLOR_FormatYUV420Planar: return "OMX_COLOR_FormatYUV420Planar";
  case OMX_COLOR_FormatYUV420PackedPlanar: return "OMX_COLOR_FormatYUV420PackedPlanar";
  case OMX_COLOR_FormatYUV420SemiPlanar: return "OMX_COLOR_FormatYUV420SemiPlanar";
  case OMX_COLOR_FormatYUV422Planar: return "OMX_COLOR_FormatYUV422Planar";
  case OMX_COLOR_FormatYUV422PackedPlanar: return "OMX_COLOR_FormatYUV422PackedPlanar";
  case OMX_COLOR_FormatYUV422SemiPlanar: return "OMX_COLOR_FormatYUV422SemiPlanar";
  case OMX_COLOR_FormatYCbYCr: return "OMX_COLOR_FormatYCbYCr";
  case OMX_COLOR_FormatYCrYCb: return "OMX_COLOR_FormatYCrYCb";
  case OMX_COLOR_FormatCbYCrY: return "OMX_COLOR_FormatCbYCrY";
  case OMX_COLOR_FormatCrYCbY: return "OMX_COLOR_FormatCrYCbY";
  case OMX_COLOR_FormatYUV444Interleaved: return "OMX_COLOR_FormatYUV444Interleaved";
  case OMX_COLOR_FormatRawBayer8bit: return "OMX_COLOR_FormatRawBayer8bit";
  case OMX_COLOR_FormatRawBayer10bit: return "OMX_COLOR_FormatRawBayer10bit";
  case OMX_COLOR_FormatRawBayer8bitcompressed: return "OMX_COLOR_FormatRawBayer8bitcompressed";
  case OMX_COLOR_FormatL2: return "OMX_COLOR_FormatL2"; 
  case OMX_COLOR_FormatL4: return "OMX_COLOR_FormatL4"; 
  case OMX_COLOR_FormatL8: return "OMX_COLOR_FormatL8"; 
  case OMX_COLOR_FormatL16: return "OMX_COLOR_FormatL16"; 
  case OMX_COLOR_FormatL24: return "OMX_COLOR_FormatL24"; 
  case OMX_COLOR_FormatL32: return "OMX_COLOR_FormatL32";
  case OMX_COLOR_FormatYUV420PackedSemiPlanar: return "OMX_COLOR_FormatYUV420PackedSemiPlanar";
  case OMX_COLOR_FormatYUV422PackedSemiPlanar: return "OMX_COLOR_FormatYUV422PackedSemiPlanar";
  case OMX_COLOR_Format18BitBGR666: return "OMX_COLOR_Format18BitBGR666";
  case OMX_COLOR_Format24BitARGB6666: return "OMX_COLOR_Format24BitARGB6666";
  case OMX_COLOR_Format24BitABGR6666: return "OMX_COLOR_Format24BitABGR6666";

  case OMX_COLOR_FormatAndroidOpaque: return "OMX_COLOR_FormatAndroidOpaque";
  case OMX_TI_COLOR_FormatYUV420PackedSemiPlanar: return "OMX_TI_COLOR_FormatYUV420PackedSemiPlanar";
  case OMX_QCOM_COLOR_FormatYVU420SemiPlanar: return "OMX_QCOM_COLOR_FormatYVU420SemiPlanar";
  case OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka: return "OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka";
  case OMX_SEC_COLOR_FormatNV12Tiled: return "OMX_SEC_COLOR_FormatNV12Tiled";
  case OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar32m: return "OMX_QCOM_COLOR_FormatYUV420PackedSemiPlanar32m";

  // case QOMX_COLOR_FormatYVU420SemiPlanar: return "QOMX_COLOR_FormatYVU420SemiPlanar";
  case QOMX_COLOR_FormatYVU420PackedSemiPlanar32m4ka: return "QOMX_COLOR_FormatYVU420PackedSemiPlanar32m4ka";
  case QOMX_COLOR_FormatYUV420PackedSemiPlanar16m2ka: return "QOMX_COLOR_FormatYUV420PackedSemiPlanar16m2ka";
  // case QOMX_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka: return "QOMX_COLOR_FormatYUV420PackedSemiPlanar64x32Tile2m8ka";
  // case QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m: return "QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m";
  case QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mMultiView: return "QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mMultiView";
  case QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mCompressed: return "QOMX_COLOR_FORMATYUV420PackedSemiPlanar32mCompressed";
  case QOMX_COLOR_Format32bitRGBA8888: return "QOMX_COLOR_Format32bitRGBA8888";
  case QOMX_COLOR_Format32bitRGBA8888Compressed: return "QOMX_COLOR_Format32bitRGBA8888Compressed";

  default:
    return "unkn";
  }
}


void encoder_init(EncoderState *s, const char* filename, int width, int height, int fps, int bitrate) {
  int err;

  memset(s, 0, sizeof(*s));
  s->filename = filename;
  s->width = width;
  s->height = height;
  s->fps = fps;
  mutex_init_reentrant(&s->lock);

  s->segment = -1;

  s->state = OMX_StateLoaded;

  s->codec_config = NULL;

  queue_init(&s->free_in);
  queue_init(&s->done_out);

  pthread_mutex_init(&s->state_lock, NULL);
  pthread_cond_init(&s->state_cv, NULL);

  err = OMX_GetHandle(&s->handle, (OMX_STRING)"OMX.qcom.video.encoder.hevc",
                      s, &omx_callbacks);
  // err = OMX_GetHandle(&s->handle, (OMX_STRING)"OMX.qcom.video.encoder.avc",
  //                     s, &omx_callbacks);
  assert(err == OMX_ErrorNone);
  // printf("handle: %p\n", s->handle);

  // setup input port

  OMX_PARAM_PORTDEFINITIONTYPE in_port = {0};
  in_port.nSize = sizeof(in_port);
  in_port.nPortIndex = (OMX_U32) PORT_INDEX_IN;
  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);


  in_port.format.video.nFrameWidth = s->width;
  in_port.format.video.nFrameHeight = s->height;
  in_port.format.video.nStride = VENUS_Y_STRIDE(COLOR_FMT_NV12, s->width);
  in_port.format.video.nSliceHeight = s->height;
  // in_port.nBufferSize = (s->width * s->height * 3) / 2;
  in_port.nBufferSize = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, s->width, s->height);
  in_port.format.video.xFramerate = (s->fps * 65536);
  in_port.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;
  // in_port.format.video.eColorFormat = OMX_COLOR_FormatYUV420SemiPlanar;
  in_port.format.video.eColorFormat = QOMX_COLOR_FORMATYUV420PackedSemiPlanar32m;

  err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);


  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &in_port);
  assert(err == OMX_ErrorNone);
  s->num_in_bufs = in_port.nBufferCountActual;

  // printf("in width: %d, stride: %d\n",
  //   in_port.format.video.nFrameWidth, in_port.format.video.nStride);

  // setup output port

  OMX_PARAM_PORTDEFINITIONTYPE out_port = {0};
  out_port.nSize = sizeof(out_port);
  out_port.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR)&out_port);
  assert(err == OMX_ErrorNone);

  out_port.format.video.nFrameWidth = s->width;
  out_port.format.video.nFrameHeight = s->height;
  out_port.format.video.xFramerate = 0;
  out_port.format.video.nBitrate = bitrate;
  out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingHEVC;
  // out_port.format.video.eCompressionFormat = OMX_VIDEO_CodingAVC;
  out_port.format.video.eColorFormat = OMX_COLOR_FormatUnused;

  err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &out_port);
  assert(err == OMX_ErrorNone);


  err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition,
                         (OMX_PTR) &out_port);
  assert(err == OMX_ErrorNone);
  s->num_out_bufs = out_port.nBufferCountActual;

  // printf("out buf num %d\n", out_port.nBufferSize);


  // setup HEVC

  OMX_VIDEO_PARAM_HEVCTYPE hecv_type = {0};
  hecv_type.nSize = sizeof(hecv_type);
  hecv_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(s->handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc,
                         (OMX_PTR) &hecv_type);
  assert(err == OMX_ErrorNone);

  hecv_type.eProfile = OMX_VIDEO_HEVCProfileMain;
  hecv_type.eLevel = OMX_VIDEO_HEVCHighTierLevel5;

  err = OMX_SetParameter(s->handle, (OMX_INDEXTYPE)OMX_IndexParamVideoHevc,
                         (OMX_PTR) &hecv_type);
  assert(err == OMX_ErrorNone);


  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(s->handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);

  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = bitrate;

  err = OMX_SetParameter(s->handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);


#if 0
    // setup h264
  OMX_VIDEO_PARAM_BITRATETYPE bitrate_type = {0};
  bitrate_type.nSize = sizeof(bitrate_type);
  bitrate_type.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(s->handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);

  bitrate_type.eControlRate = OMX_Video_ControlRateVariable;
  bitrate_type.nTargetBitrate = 2000000;

  err = OMX_SetParameter(s->handle, OMX_IndexParamVideoBitrate,
                         (OMX_PTR) &bitrate_type);
  assert(err == OMX_ErrorNone);


  OMX_VIDEO_PARAM_AVCTYPE avc = { 0 };
  avc.nSize = sizeof(avc);
  avc.nPortIndex = (OMX_U32) PORT_INDEX_OUT;
  err = OMX_GetParameter(s->handle, OMX_IndexParamVideoAvc, &avc);
  assert(err == OMX_ErrorNone);

  avc.nBFrames = 0;
  avc.nPFrames = 20;//avctx->gop_size - 1;

  avc.eProfile = OMX_VIDEO_AVCProfileBaseline;
  avc.eLevel = OMX_VIDEO_AVCLevel31;

  avc.nAllowedPictureTypes |= OMX_VIDEO_PictureTypeB;
  avc.eLoopFilterMode = OMX_VIDEO_AVCLoopFilterEnable;

  err = OMX_SetParameter(s->handle, OMX_IndexParamVideoAvc, &avc);
  assert(err == OMX_ErrorNone);
#endif


  // for (int i = 0; ; i++) {
  //   OMX_VIDEO_PARAM_PORTFORMATTYPE video_port_format = {0};
  //   video_port_format.nSize = sizeof(video_port_format);
  //   video_port_format.nIndex = i;
  //   video_port_format.nPortIndex = PORT_INDEX_IN;
  //   if (OMX_GetParameter(s->handle, OMX_IndexParamVideoPortFormat, &video_port_format) != OMX_ErrorNone)
  //       break;
  //   printf("in %d: compression 0x%x format 0x%x %s\n", i,
  //          video_port_format.eCompressionFormat, video_port_format.eColorFormat,
  //          omx_color_fomat_name(video_port_format.eColorFormat));
  // }

  // for (int i=0; i<32; i++) {
  //   OMX_VIDEO_PARAM_PROFILELEVELTYPE params = {0};
  //   params.nSize = sizeof(params);
  //   params.nPortIndex = PORT_INDEX_OUT;
  //   params.nProfileIndex = i;
  //   if (OMX_GetParameter(s->handle, OMX_IndexParamVideoProfileLevelQuerySupported, &params) != OMX_ErrorNone)
  //       break;
  //   printf("profile %d level 0x%x\n", params.eProfile, params.eLevel);
  // }

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);

  s->in_buf_headers = calloc(s->num_in_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<s->num_in_bufs; i++) {
    err = OMX_AllocateBuffer(s->handle, &s->in_buf_headers[i], PORT_INDEX_IN, s,
                             in_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  s->out_buf_headers = calloc(s->num_out_bufs, sizeof(OMX_BUFFERHEADERTYPE*));
  for (int i=0; i<s->num_out_bufs; i++) {
    err = OMX_AllocateBuffer(s->handle, &s->out_buf_headers[i], PORT_INDEX_OUT, s,
                             out_port.nBufferSize);
    assert(err == OMX_ErrorNone);
  }

  wait_for_state(s, OMX_StateIdle);

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateExecuting, NULL);
  assert(err == OMX_ErrorNone);

  wait_for_state(s, OMX_StateExecuting);

  // give omx all the output buffers
  for (int i = 0; i < s->num_out_bufs; i++) {
    // printf("fill %p\n", s->out_buf_headers[i]);
    err = OMX_FillThisBuffer(s->handle, s->out_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }

  // fill the input free queue
  for (int i = 0; i < s->num_in_bufs; i++) {
    queue_push(&s->free_in, (void*)s->in_buf_headers[i]);
  }
}

static void handle_out_buf(EncoderState *s, OMX_BUFFERHEADERTYPE *out_buf) {
  int err;

  uint8_t *buf_data = out_buf->pBuffer + out_buf->nOffset;

  if (out_buf->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {
    if (s->codec_config_len < out_buf->nFilledLen) {
      s->codec_config = realloc(s->codec_config, out_buf->nFilledLen);
    }
    s->codec_config_len = out_buf->nFilledLen;
    memcpy(s->codec_config, buf_data, out_buf->nFilledLen);
  }

  if (s->stream_sock_raw) {
    uint64_t current_time = nanos_since_boot();
    uint64_t diff = current_time - out_buf->nTimeStamp*1000LL;
    double msdiff = (double) diff / 1000000.0;
    // printf("encoded latency to tsEof: %f\n", msdiff);
    zmq_send(s->stream_sock_raw, &out_buf->nTimeStamp, sizeof(out_buf->nTimeStamp), ZMQ_SNDMORE);
    zmq_send(s->stream_sock_raw, buf_data, out_buf->nFilledLen, 0);
  }


  if (s->of) {
    // printf("write %d flags 0x%x\n", out_buf->nFilledLen, out_buf->nFlags);
    fwrite(buf_data, out_buf->nFilledLen, 1, s->of);
    fwrite(&out_buf->nFilledLen, sizeof(uint32_t),1,s->frame_size);
  }

  // give omx back the buffer
  err = OMX_FillThisBuffer(s->handle, out_buf);
  assert(err == OMX_ErrorNone);
}

int encoder_encode_frame(EncoderState *s, uint64_t ts,
                         const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                         int *frame_segment, VIPCBufExtra *extra) {
  int err;

  pthread_mutex_lock(&s->lock);

  if (s->opening) {
    encoder_open(s, s->next_path);
    s->opening = false;
  }

  if (!s->open) {
    pthread_mutex_unlock(&s->lock);
    return -1;
  }

  // this sometimes freezes... put it outside the encoder lock so we can still trigger rotates...
  // THIS IS A REALLY BAD IDEA, but apparently the race has to happen 30 times to trigger this
  pthread_mutex_unlock(&s->lock);
  OMX_BUFFERHEADERTYPE* in_buf = queue_pop(&s->free_in);
  pthread_mutex_lock(&s->lock);
  //TODO: alttaki satir debugdan sonra silinmeli
  //s->rotating = false;
  if (s->rotating) {
    encoder_close(s);
    encoder_open(s, s->next_path);
    s->segment = s->next_segment;
    s->rotating = false;
  }

  int ret = s->counter;

  uint8_t *in_buf_ptr = in_buf->pBuffer;
  // printf("in_buf ptr %p\n", in_buf_ptr);

  uint8_t *in_y_ptr = in_buf_ptr;
  int in_y_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, s->width);
  int in_uv_stride = VENUS_UV_STRIDE(COLOR_FMT_NV12, s->width); 
  // uint8_t *in_uv_ptr = in_buf_ptr + (s->width * s->height);
  uint8_t *in_uv_ptr = in_buf_ptr + (in_y_stride * VENUS_Y_SCANLINES(COLOR_FMT_NV12, s->height));

  err = I420ToNV12(y_ptr, s->width,
                   u_ptr, s->width/2,
                   v_ptr, s->width/2,
                   in_y_ptr, in_y_stride,
                   in_uv_ptr, in_uv_stride,
                   s->width, s->height);
  assert(err == 0);

  // in_buf->nFilledLen = (s->width*s->height) + (s->width*s->height/2);
  in_buf->nFilledLen = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, s->width, s->height);
  in_buf->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
  in_buf->nOffset = 0;
  in_buf->nTimeStamp = extra->timestamp_eof/1000LL;

  err = OMX_EmptyThisBuffer(s->handle, in_buf);
  assert(err == OMX_ErrorNone);

  // pump output
  while (true) {
    OMX_BUFFERHEADERTYPE *out_buf = queue_try_pop(&s->done_out);
    if (!out_buf) {
      break;
    }
    handle_out_buf(s, out_buf);
  }

  s->dirty = true;

  s->counter++;

  if (frame_segment) {
    *frame_segment = s->segment;
  }

  if (s->closing) {
    encoder_close(s);
    s->closing = false;
  }

  pthread_mutex_unlock(&s->lock);
  return ret;
}

void encoder_open(EncoderState *s, const char* path) {
  int err;

  pthread_mutex_lock(&s->lock);

  snprintf(s->vid_path, sizeof(s->vid_path), "%s/%s.hevc", path, s->filename);
  s->of = fopen("/sdcard/surus.hevc", "ab");
  s->frame_size = fopen("/sdcard/sizes.txt","ab");
  printf("aaaaaa Dosyalar acildi!!!!!!!! - codec_config_len = %d", s->codec_config_len);
  assert(s->of);

  if (s->codec_config_len > 0) {
    fwrite(s->codec_config, s->codec_config_len, 1, s->of);
  }

  // create camera lock file
  snprintf(s->lock_path, sizeof(s->lock_path), "%s/%s.lock", path, s->filename);
  int lock_fd = open(s->lock_path, O_RDWR | O_CREAT, 0777);
  assert(lock_fd >= 0);
  close(lock_fd);

  s->open = true;
  s->counter = 0;

  pthread_mutex_unlock(&s->lock);
}

void encoder_close(EncoderState *s) {
  int err;

  pthread_mutex_lock(&s->lock);

  if (s->open) {
    if (s->dirty) {
      // drain output only if there could be frames in the encoder

      OMX_BUFFERHEADERTYPE* in_buf = queue_pop(&s->free_in);
      in_buf->nFilledLen = 0;
      in_buf->nOffset = 0;
      in_buf->nFlags = OMX_BUFFERFLAG_EOS;
      in_buf->nTimeStamp = 0;

      err = OMX_EmptyThisBuffer(s->handle, in_buf);
      assert(err == OMX_ErrorNone);

      while (true) {
        OMX_BUFFERHEADERTYPE *out_buf = queue_pop(&s->done_out);

        handle_out_buf(s, out_buf);

        if (out_buf->nFlags & OMX_BUFFERFLAG_EOS) {
          break;
        }
      }
      s->dirty = false;
    }
    printf("aaaaaa Dosyar kapandi!!!!!!!! ");
    fclose(s->of);
    fclose(s->frame_size);
    unlink(s->lock_path);
  }
  s->open = false;

  pthread_mutex_unlock(&s->lock);
}

void encoder_rotate(EncoderState *s, const char* new_path, int new_segment) {
  pthread_mutex_lock(&s->lock);
  snprintf(s->next_path, sizeof(s->next_path), "%s", new_path);
  s->next_segment = new_segment;
  printf("aaaaaa rotateye girdi? next_segment - open: !!!!!!!! %d - %d",s->next_segment,s->open);
  if (s->open) {
    if (s->next_segment == -1) {
      s->closing = true;
    } else {
      s->rotating = true;
    }
  } else {
    s->segment = s->next_segment;
    s->opening = true;
  }
  pthread_mutex_unlock(&s->lock);
}

void encoder_destroy(EncoderState *s) {
  int err;

  assert(!s->open);

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
  assert(err == OMX_ErrorNone);

  wait_for_state(s, OMX_StateIdle);

  err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateLoaded, NULL);
  assert(err == OMX_ErrorNone);

  for (int i=0; i<s->num_in_bufs; i++) {
    err = OMX_FreeBuffer(s->handle, PORT_INDEX_IN, s->in_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }
  free(s->in_buf_headers);

  for (int i=0; i<s->num_out_bufs; i++) {
    err = OMX_FreeBuffer(s->handle, PORT_INDEX_OUT, s->out_buf_headers[i]);
    assert(err == OMX_ErrorNone);
  }
  free(s->out_buf_headers);

  wait_for_state(s, OMX_StateLoaded);

  err = OMX_FreeHandle(s->handle);
  assert(err == OMX_ErrorNone);
}

#if 0

// cd one/selfdrive/visiond
// clang
//   -fPIC -pie
//   -std=gnu11 -O2 -g
//   -o encoder
//   -I ~/one/selfdrive
//   -I ~/one/phonelibs/openmax/include
//   -I ~/one/phonelibs/libyuv/include
//   -lOmxVenc -lOmxCore -llog
//   encoder.c
//   buffering.c
//   -L ~/one/phonelibs/libyuv/lib -l:libyuv.a

int main() {
  int err;

  EncoderState state;
  EncoderState *s = &state;
  memset(s, 0, sizeof(*s));

  int w = 1164;
  int h = 874;

  encoder_init(s, w, h, 20);
  printf("inited\n");

  encoder_open(s, "/sdcard/t1");

  // uint8_t *tmpy = malloc(640*480);
  // uint8_t *tmpu = malloc((640/2)*(480/2));
  // uint8_t *tmpv = malloc((640/2)*(480/2));

  // memset(tmpy, 0, 640*480);
  // // memset(tmpu, 0xff, (640/2)*(480/2));
  // memset(tmpv, 0, (640/2)*(480/2));

// #if 0
  // FILE *infile = fopen("/sdcard/camera_t2.yuv", "rb");
  uint8_t *inbuf = malloc(w*h*3/2);
  memset(inbuf, 0, w*h*3/2);

  for (int i=0; i<20*3+5; i++) {
    
    // fread(inbuf, w*h*3/2, 1, infile);

    uint8_t *tmpy = inbuf;
    uint8_t *tmpu = inbuf + w*h;
    uint8_t *tmpv = inbuf + w*h + (w/2)*(h/2);

    for (int y=0; y<h/2; y++) {
      for (int x=0; x<w/2; x++) {
        tmpu[y * (w/2) + x] = (i ^ y ^ x);
      }
    }

    encoder_encode_frame(s, 20000*i, tmpy, tmpu, tmpv);
  }
// #endif

  // while(1);

  printf("done\n");

  // encoder_close(s);

  // printf("restart\n");
  // fclose(s->of);
  // s->of = fopen("/sdcard/tmpout2.hevc", "wb");
  // if (s->codec_config) {
  //   fwrite(s->codec_config, s->codec_config_len, 1, s->of);
  // }
  // encoder_open(s, "/sdcard/t1");

  encoder_rotate(s, "/sdcard/t2");

  for (int i=0; i<20*3+5; i++) {
    
    // fread(inbuf, w*h*3/2, 1, infile);

    uint8_t *tmpy = inbuf;
    uint8_t *tmpu = inbuf + w*h;
    uint8_t *tmpv = inbuf + w*h + (w/2)*(h/2);

    for (int y=0; y<h/2; y++) {
      for (int x=0; x<w/2; x++) {
        tmpu[y * (w/2) + x] = (i ^ y ^ x);
      }
    }

    encoder_encode_frame(s, 20000*i, tmpy, tmpu, tmpv);
  }
  encoder_close(s);

  return 0;
}
#endif
