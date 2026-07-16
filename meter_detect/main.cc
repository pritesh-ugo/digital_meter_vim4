/****************************************************************************
*
* Copyright (c) 2019  by amlogic Corp.  All rights reserved.
*
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn_sdk.h"
#include "nn_util.h"
#include "postprocess_util.h"
#include "nn_demo.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <getopt.h>
#include <chrono> // Added for inference timing
#include <sys/stat.h>
#include <sys/types.h>
#include <string>

#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640

struct option longopts[] = {
    { "path",           required_argument,  NULL,   'p' },
    { "model",          required_argument,  NULL,   'm' },
    { "help",           no_argument,        NULL,   'H' },
    { 0, 0, 0, 0 }
};

nn_input inData;
cv::Mat img;
aml_module_t modelType;

static int input_width, input_high, input_channel;
static const char *kDebugDir = "../debug";
static const char *kResultsDir = "../results";

// UPDATE THIS LIST TO MATCH YOUR CUSTOM MODEL'S CLASSES
static const char *coco_names[] = {
    "meter_screen" 
};

typedef enum _amlnn_detect_type_ {
    Accuracy_Detect_Yolo_V3 = 0
} amlnn_detect_type;

static void ensure_directory(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

static void ensure_output_dirs() {
    ensure_directory(kDebugDir);
    ensure_directory(kResultsDir);
}

void* init_network_file(const char *mpath) {
    void *qcontext = NULL;
    aml_config config;
    memset(&config, 0, sizeof(aml_config));

    config.nbgType = NN_ADLA_FILE;
    config.path = mpath;
    config.modelType = ADLA_LOADABLE;
    config.typeSize = sizeof(aml_config);

    qcontext = aml_module_create(&config);
    if (qcontext == NULL) {
        printf("amlnn_init is fail\n");
        return NULL;
    }

    if (config.nbgType == NN_ADLA_MEMORY && config.pdata != NULL) {
        free((void*)config.pdata);
    }

    return qcontext;
}

int set_input(void *qcontext, const char *jpath, double *preprocess_ms) {
    int ret = 0;
    cv::Mat temp_img, rgb_img;
    auto start_pre = std::chrono::high_resolution_clock::now();
    
    img = cv::imread(jpath);
    if (img.empty()) {
        printf("Error: Could not open or find the image at %s\n", jpath);
        return -1;
    }
    
    // 1. Resize the image to model dimensions
    cv::resize(img, temp_img, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));

    // SAVE POINT 1: Save the 640x640 intermediate image
    cv::imwrite("1_input_640x640.jpg", temp_img);

    // 2. Convert BGR to RGB (OpenCV loads BGR, YOLO expects RGB)
    cv::cvtColor(temp_img, rgb_img, cv::COLOR_BGR2RGB);

    // 3. Hand the RGB memory directly to the NPU
    inData.input_type = BINARY_RAW_DATA;
    inData.input = rgb_img.data;
    inData.input_index = 0;
    inData.size = MODEL_WIDTH * MODEL_HEIGHT * 3; // 1,228,800 bytes
    
    ret = aml_module_input_set(qcontext, &inData);

    auto end_pre = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pre_time = end_pre - start_pre;
    if (preprocess_ms != NULL) {
        *preprocess_ms = pre_time.count();
    }

    return ret;
}

static cv::Scalar obj_id_to_color(int obj_id) {
    int const colors[6][3] = { { 1,0,1 }, { 0,0,1 }, { 0,1,1 }, { 0,1,0 }, { 1,1,0 }, { 1,0,0 } };
    int const offset = obj_id * 123457 % 6;
    int const color_scale = 150 + (obj_id * 123457) % 100;
    cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
    color *= color_scale;
    return color;
}

int run_network(void *qcontext, double preprocess_ms) {
    int ret = 0;
    nn_output *outdata = NULL;
    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));

    outconfig.format = AML_OUTDATA_FLOAT32;
    outconfig.typeSize = sizeof(aml_output_config_t);

    obj_detect_out_t yolov3_detect_out;
    
    // --- START NPU TIMER ---
    auto start_npu = std::chrono::high_resolution_clock::now();

    outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);
    if (outdata == NULL) {
        printf("aml_module_output_get error\n");
        return -1;
    }

    // --- STOP NPU TIMER ---
    auto end_npu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> npu_time = end_npu - start_npu;
    
    // --- START CPU POST-PROCESSING TIMER ---
    auto start_post = std::chrono::high_resolution_clock::now();

    postprocess_yolov3(outdata, &yolov3_detect_out);

    // --- STOP CPU TIMER ---
    auto end_post = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> post_time = end_post - start_post;

    // --- PRINT PROFILING RESULTS ---
    printf("\n================ TIMING PROFILING ================\n");
    printf("NPU Inference Time     : %.2f ms (%.1f FPS)\n", npu_time.count(), 1000.0 / npu_time.count());
    printf("CPU Post-Process Time  : %.2f ms\n", post_time.count());
    printf("==================================================\n\n");

    int classid = 0;
    float prob = 0;
    int left = 0, right = 0, top = 0, bot = 0;
    cv::Point pt1;
    cv::Point pt2;
    int baseline;

    printf("object_num:%d\n", yolov3_detect_out.detNum);

    for (int i = 0; i < yolov3_detect_out.detNum; i++) {
        classid = (int)yolov3_detect_out.pBox[i].objectClass;
        prob = yolov3_detect_out.pBox[i].score;
        
        left  = (yolov3_detect_out.pBox[i].x - yolov3_detect_out.pBox[i].w/2.) * img.cols;
        right = (yolov3_detect_out.pBox[i].x + yolov3_detect_out.pBox[i].w/2.) * img.cols;
        top   = (yolov3_detect_out.pBox[i].y - yolov3_detect_out.pBox[i].h/2.) * img.rows;
        bot   = (yolov3_detect_out.pBox[i].y + yolov3_detect_out.pBox[i].h/2.) * img.rows;
        
        // Safety checks to prevent OpenCV from crashing if boxes exceed image bounds
        if (left < 0) left = 0;
        if (right > img.cols - 1) right = img.cols - 1;
        if (top < 0) top = 0;
        if (bot > img.rows - 1) bot = img.rows - 1;

        if (left > right) {
            int tmp = left;
            left = right;
            right = tmp;
        }

        if (top > bot) {
            int tmp = top;
            top = bot;
            bot = tmp;
        }
        
        printf("class:%s,label_num:%d,prob:%f,left:%d,top:%d,right:%d,bot:%d\n", coco_names[classid], classid, prob, left, top, right, bot);

        // SAVE POINT 2: Extract and save the cropped meter image
        if (right > left && bot > top) {
            cv::Rect roi(left, top, right - left, bot - top);
            cv::Mat crop = img(roi);
            char crop_filename[64];
            snprintf(crop_filename, sizeof(crop_filename), "%s/meter_crop_%d.jpg", kDebugDir, i);
            cv::imwrite(crop_filename, crop);
        }

        // Draw bounding box and label on the main image
        pt1 = cv::Point(left, top);
        pt2 = cv::Point(right, bot);
        cv::Rect rect(left, top, right-left, bot-top);
        cv::rectangle(img, rect, obj_id_to_color(classid), 2, 8, 0);

        if (top < 50) {
            top = 50;
            left += 10;
        }
        cv::Size text_size = cv::getTextSize(coco_names[classid], cv::FONT_HERSHEY_COMPLEX, 0.5 , 1, &baseline);
        cv::Rect rect1(left, top-20, text_size.width+10, 20);
        cv::rectangle(img, rect1, obj_id_to_color(classid), -1);
        cv::putText(img, coco_names[classid], cvPoint(left+5, top-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
    
    // Save the full annotated meter detection result.
    cv::imwrite("../results/detected_meter.jpg", img);
    cv::imwrite("../results/detected_meter_annotated.jpg", img);

    printf("\n================ STAGE PROFILING ================\n");
    printf("Pre-process Time       : %.2f ms\n", preprocess_ms);
    printf("NPU Inference Time     : %.2f ms (%.1f FPS)\n", npu_time.count(), 1000.0 / npu_time.count());
    printf("CPU Post-Process Time  : %.2f ms\n", post_time.count());
    printf("==================================================\n\n");
    
    return ret;
}

int destroy_network(void *qcontext) {
    int ret = aml_module_destroy(qcontext);
    return ret;
}

int main(int argc, char **argv) {
    int c;
    int ret = 0;
    void *context = NULL;
    char *model_path = NULL;
    char *input_data = NULL;
    double preprocess_time_ms = 0.0;
    
    input_width = MODEL_WIDTH;
    input_high = MODEL_HEIGHT;
    input_channel = 3;

    while ((c = getopt_long(argc, argv, "p:m:H", longopts, NULL)) != -1) {
        switch (c) {
            case 'p':
                input_data = optarg;
                break;
            case 'm':
                model_path = optarg;
                break;
            default:
                printf("%s [-p picture path] [-m model path]  [-H]\n", argv[0]);
                exit(1);
        }
    }

    if (!model_path || !input_data) {
        printf("Error: Missing model path (-m) or input image path (-p)\n");
        exit(1);
    }

    ensure_output_dirs();

    context = init_network_file(model_path);
    if (context == NULL) {
        printf("init_network fail.\n");
        return -1;
    }

    ret = set_input(context, input_data, &preprocess_time_ms);
    if (ret != 0) {
        printf("set_input fail.\n");
        return -1;
    }

    ret = run_network(context, preprocess_time_ms);
    if (ret != 0) {
        printf("run_network fail.\n");
        return -1;
    }

    ret = destroy_network(context);
    if (ret != 0) {
        printf("destroy_network fail.\n");
        return -1;
    }

    return ret;
}
