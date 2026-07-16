/****************************************************************************
*   amlogic nn api util header file
*
*   Neural Network appliction network definition some util header file
*
*   Date: 2019.8
****************************************************************************/
#ifndef _AMLOGIC_NN_DEMO_H
#define _AMLOGIC_NN_DEMO_H

#include "nn_sdk.h"
#include "nn_util.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct __nn_image_classify
{
	float	score[5];
	unsigned int  topClass[5];
}img_classify_out_t;

typedef struct __nn_obj_detect
{
	unsigned int  detNum;
	detBox *pBox;
}obj_detect_out_t;

void activate_array(float *start, int num);
int entry_index(int lw, int lh, int lclasses, int loutputs, int batch, int location, int entry);
void softmax(float *input, int n, float temp, float *output);
void flatten(float *x, int size, int layers, int batch, int forward);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
int nms_comparator(const void *pa, const void *pb);
float box_iou(box a, box b);
float box_union(box a, box b);
float box_intersection(box a, box b);
float overlap(float x1, float w1, float x2, float w2);
float logistic_activate(float x);
float sigmod(float x);
unsigned char *transpose(const unsigned char * src,int width,int height);
void *post_process_all_module(aml_module_t type,nn_output *pOut);
int max_index(float *a, int n);
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h);
void process_top5(float *buf,unsigned int num,img_classify_out_t* clsout);
void* postprocess_object_detect(nn_output *pout);
unsigned char *get_jpeg_rawData(const char *name,unsigned int width,unsigned int height);
	
#ifdef __cplusplus
} //extern "C"
#endif
#endif