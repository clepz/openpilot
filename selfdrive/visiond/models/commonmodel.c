#include "commonmodel.h"

#include <czmq.h>
#include "cereal/gen/c/log.capnp.h"
#include "common/mat.h"
#include "common/timing.h"

void model_input_init(ModelInput* s, int width, int height,
                      cl_device_id device_id, cl_context context) {
  int err;
  s->device_id = device_id;
  s->context = context;

  transform_init(&s->transform, context, device_id);
  s->transformed_width = width;
  s->transformed_height = height;

  s->transformed_y_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       s->transformed_width*s->transformed_height, NULL, &err);
  assert(err == 0);
  s->transformed_u_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       (s->transformed_width/2)*(s->transformed_height/2), NULL, &err);
  assert(err == 0);
  s->transformed_v_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       (s->transformed_width/2)*(s->transformed_height/2), NULL, &err);
  assert(err == 0);

  s->net_input_size = ((width*height*3)/2)*sizeof(float);
  s->net_input = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                s->net_input_size, (void*)NULL, &err);
  assert(err == 0);

  loadyuv_init(&s->loadyuv, context, device_id, s->transformed_width, s->transformed_height);
}

float *model_input_prepare(ModelInput* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform) {
  int err;
  int i = 0;
  transform_queue(&s->transform, q,
                  yuv_cl, width, height,
                  s->transformed_y_cl, s->transformed_u_cl, s->transformed_v_cl,
                  s->transformed_width, s->transformed_height,
                  transform);


  //----------------
  float *yuv_cl_y = (float *)clEnqueueMapBuffer(q, s->transformed_y_cl, CL_TRUE,
                                            CL_MAP_READ, 0, width*height,
                                            0, NULL, NULL, &err);
  FILE *yuv_cl_y_file = fopen("/sdcard/yuv_cl_y.y", "wb");
    fwrite(yuv_cl_y, width*height, sizeof(float), yuv_cl_y_file);
    fclose(yuv_cl_y_file);

  //----------------
  float *yuv_cl_u = (float *)clEnqueueMapBuffer(q, s->transformed_u_cl, CL_TRUE,
                                            CL_MAP_READ, 0, width*height/4,
                                            0, NULL, NULL, &err);
  FILE *yuv_cl_u_file = fopen("/sdcard/yuv_cl_u.u", "wb");
    fwrite(yuv_cl_u, width*height/4, sizeof(float), yuv_cl_u_file);
    fclose(yuv_cl_u_file);

  //----------------
  float *yuv_cl_v = (float *)clEnqueueMapBuffer(q, s->transformed_v_cl, CL_TRUE,
                                            CL_MAP_READ, 0, width*height/4,
                                            0, NULL, NULL, &err);
  FILE *yuv_cl_v_file = fopen("/sdcard/yuv_cl_v.v", "wb");
    fwrite(yuv_cl_v, width*height/4, sizeof(float), yuv_cl_v_file);
    fclose(yuv_cl_v_file);
  //----------------
  loadyuv_queue(&s->loadyuv, q,
                s->transformed_y_cl, s->transformed_u_cl, s->transformed_v_cl,
                s->net_input);
  float *net_input_buf = (float *)clEnqueueMapBuffer(q, s->net_input, CL_TRUE,
                                            CL_MAP_READ, 0, s->net_input_size,
                                            0, NULL, NULL, &err);
  printf("commonmodel s->net_input_size: %d\n", s->net_input_size);



  clFinish(q);
  return net_input_buf;
}

void model_input_free(ModelInput* s) {
  transform_destroy(&s->transform);
  loadyuv_destroy(&s->loadyuv);
}


float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}

float softplus(float input) {
  return log1p(expf(input));
}
