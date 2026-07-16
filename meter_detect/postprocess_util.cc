/****************************************************************************
*
* Copyright (c) 2019  by amlogic Corp.  All rights reserved.
*
***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include "nn_util.h"
#include "nn_sdk.h"
#include "nn_demo.h"

int g_detect_number = 50; //max detect num

typedef enum {
    UTIL_FAILURE = -1,
    UTIL_SUCCESS = 0,
} nn_status_e;

void process_top5(float *buf, unsigned int num, img_classify_out_t* clsout) {
    int j = 0;
    unsigned int MaxClass[5]={0};
    float fMaxProb[5]={0.0};
    float *pfMaxProb = fMaxProb;
    unsigned int  *pMaxClass = MaxClass,i = 0;

    for (j = 0; j < 5; j++) {
        for (i=0; i<num; i++) {
            if ((i == *(pMaxClass+0)) || (i == *(pMaxClass+1)) || (i == *(pMaxClass+2)) ||
                (i == *(pMaxClass+3)) || (i == *(pMaxClass+4))) {
                continue;
            }
            if (buf[i] > *(pfMaxProb+j)) {
                *(pfMaxProb+j) = buf[i];
                *(pMaxClass+j) = i;
            }
        }
    }
    for (i=0; i<5; i++) {
        if (clsout == NULL) {
            printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
        } else {
            clsout->score[i] = fMaxProb[i];
            clsout->topClass[i] = MaxClass[i];
        }
    }
}

float logistic_activate(float x) {
    return 1./(1. + exp(-x));
}

// REVERTED TO NHWC (INTERLEAVED) MEMORY LAYOUT
box get_region_box(float *x, int box_index, int i, int j, int w, int h) {
    box b;
    float tmp[4] = {0};
    for (int k = 0; k < 4; k++) {
        float sum = 0;
        float exps[16] = {0};
        for(int m = 0; m < 16; m++) {
            exps[m] = exp(x[box_index + k * 16 + m]);
            sum += exps[m];
        }
        for(int m = 0; m < 16; m++) {
            tmp[k] += m * exps[m] / sum;
        }
    }
    b.x = (j + 0.5 - tmp[0]) / w;
    b.y = (i + 0.5 - tmp[1]) / h;
    b.w = (j + 0.5 + tmp[2]) / w;
    b.h = (i + 0.5 + tmp[3]) / h;
    b.w = b.w - b.x;
    b.h = b.h - b.y;
    b.x = b.x + b.w / 2;
    b.y = b.y + b.h / 2;
    return b;
}

int max_index(float *a, int n) {
    int i, max_i = 0;
    float max = a[0];
    if (n <= 0) return -1;
    for (i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b) {
    float area = 0;
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    area = w*h;
    return area;
}

float box_union(box a, box b) {
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b) {
    return box_intersection(a, b)/box_union(a, b);
}

int nms_comparator(const void *pa, const void *pb) {
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.classId] - b.probs[b.index][b.classId];
    if (diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh) {
    int i, j, k;
    sortable_bbox *s = (sortable_bbox *)calloc(total, sizeof(sortable_bbox));
    for (i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].classId = 0;
        s[i].probs = probs;
    }
    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            s[i].classId = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for (i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0) continue;
            for (j = i+1; j < total; ++j) {
                box b = boxes[s[j].index];
                if (probs[s[j].index][k]>0) {
                    if (box_iou(boxes[s[i].index], b) > thresh) {
                        probs[s[j].index][k] = 0;
                    }
                }
            }
        }
    }
    free(s);
}

void yolov3_result(int num, float thresh, box *boxes, float **probs, int classes, obj_detect_out_t* dectout) {
    int i = 0, detect_num = 0;
    dectout->pBox = (detBox*)malloc(MAX_DETECT_NUM*sizeof(detBox));
    for (i = 0; i < num; ++i) {
        int classId = max_index(probs[i], classes);
        float prob = probs[i][classId];
        if (prob > thresh) {
            if (detect_num >= g_detect_number) {
                break;
            }
            dectout->pBox[detect_num].x = boxes[i].x;
            dectout->pBox[detect_num].y = boxes[i].y;
            dectout->pBox[detect_num].w = boxes[i].w;
            dectout->pBox[detect_num].h = boxes[i].h;
            dectout->pBox[detect_num].score = prob;
            dectout->pBox[detect_num].objectClass = (float)classId;
            detect_num++ ;
        }
    }
    dectout->detNum = detect_num;
}

// REVERTED TO NHWC (INTERLEAVED) MEMORY LAYOUT
int yolo_v3_post_process_onescale(float *predictions, int input_size[3], box *boxes, float **probs, float threshold_in, int *yolov3_box_num_after_filter) {
    int i, j, k, index;
    int num_class = 1;
    int bb_size = 64 + num_class; // 65 channels
    int modelWidth = input_size[0];
    int modelHeight = input_size[1];
    float threshold = threshold_in;
    float max_prob;

    for (j = 0; j < modelWidth*modelHeight; ++j)
        probs[j] = (float *)calloc(num_class+1, sizeof(float *));

    for (i = 0; i < modelHeight; ++i) {
        for (j = 0; j < modelWidth; ++j) {
            index = i * modelWidth + j;
            max_prob = 0;
            for (k = 0; k < num_class; ++k) {
                float prob = logistic_activate(predictions[index * bb_size + k]);
                probs[index][k] = (prob > threshold) ? prob : 0;
                max_prob = (prob > threshold && prob > max_prob) ? prob : max_prob;
            }
            int box_index = index * bb_size + num_class;
            boxes[index] = get_region_box(predictions, box_index, i, j, modelWidth, modelHeight);
            boxes[index].prob_obj = (max_prob > threshold) ? max_prob : 0;
            
            if (max_prob > threshold) {
                (*yolov3_box_num_after_filter)++;
            }
        }
    }
    return 0;
}

void yolov3_postprocess(float **predictions, int width, int height, int modelWidth, int modelHeight, int input_num, obj_detect_out_t* dectout) {
    int nn_width = 640;
    int nn_height = 640;
    int size[3] = {nn_width/32, nn_height/32, 1 + 64};
    int yolov3_box_num_after_filter = 0;

    int j, k, index;
    int num_class = 1;
    float threshold = 0.25; // Raised threshold back up!
    float iou_threshold = 0.45;

    int size2[3] = {size[0]*2, size[1]*2, size[2]};
    int size4[3] = {size[0]*4, size[1]*4, size[2]};
    int len1 = size[0]*size[1]*size[2];
    int box1 = len1/(num_class+64);

    box *boxes = (box *)calloc(box1*(1+4+16), sizeof(box));
    float **probs = (float **)calloc(box1*(1+4+16), sizeof(float *));

    yolo_v3_post_process_onescale(predictions[2], size, boxes, &probs[0], threshold, &yolov3_box_num_after_filter);
    yolo_v3_post_process_onescale(predictions[1], size2, &boxes[box1], &probs[box1], threshold, &yolov3_box_num_after_filter);
    yolo_v3_post_process_onescale(predictions[0], size4, &boxes[box1*(1+4)], &probs[box1*(1+4)], threshold, &yolov3_box_num_after_filter);

    box *tmp_boxes = (box *)calloc(yolov3_box_num_after_filter, sizeof(box));
    float **tmp_probs = (float **)calloc(yolov3_box_num_after_filter, sizeof(float *));

    for (k = 0; k < yolov3_box_num_after_filter; k++) {
        tmp_probs[k] = (float *)calloc(num_class+1, sizeof(float *));
    }
    
    for (index = 0, k = 0; index < box1*(1+4+16); index++) {
        if ((fabs(boxes[index].prob_obj)-0) > 0.000001) {
            tmp_probs[k] = probs[index];
            tmp_boxes[k] = boxes[index];
            k++;
        }
    }

    do_nms_sort(tmp_boxes, tmp_probs, yolov3_box_num_after_filter, num_class, iou_threshold);
    yolov3_result(yolov3_box_num_after_filter, threshold, tmp_boxes, tmp_probs, num_class, dectout);
    
    free(tmp_boxes);
    free(tmp_probs);
    for (j = 0; j < box1*(1+4+16); ++j) {
        free(probs[j]);
    }
    free(boxes);
    free(probs);
}

void postprocess_yolov3(nn_output *pout, obj_detect_out_t* dectout) {
    float *yolov3_buffer[3] = {NULL};
    yolov3_buffer[0] = (float*)pout->out[1].buf;// 80 80
    yolov3_buffer[1] = (float*)pout->out[0].buf;// 40 40
    yolov3_buffer[2] = (float*)pout->out[2].buf;// 20 20

    yolov3_postprocess(yolov3_buffer, 640, 640, 20, 20, 0, dectout);
}
