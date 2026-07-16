#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glob.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {

const std::string kRootDir = "/data/summerint26/models/cpy";
const std::string kMeterDetectDir = kRootDir + "/meter_detect";
const std::string kMeterCropDir = kRootDir + "/meter_crop";
const std::string kOcrDir = kRootDir + "/onnx_output";
const std::string kMeterDetectBuildDir = kMeterDetectDir + "/build_fixed";
const std::string kMeterCropBuildDir = kMeterCropDir + "/build_fixed";

const std::string kPipelineDebugDir = kRootDir + "/pipeline_debug";
const std::string kPipelineResultsDir = kRootDir + "/pipeline_results";
const std::string kPipelineStageDebugDir = kPipelineDebugDir + "/stages";

std::string shell_quote(const std::string &value) {
    std::string quoted;
    quoted.reserve(value.size() + 10);
    quoted += '\'';
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += '\'';
    return quoted;
}

void ensure_dir(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

void ensure_output_dirs() {
    ensure_dir(kPipelineDebugDir);
    ensure_dir(kPipelineResultsDir);
    ensure_dir(kPipelineStageDebugDir);
}

struct StageTiming {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
};

struct CommandResult {
    double wall_ms;
    int exit_code;
    std::string output;
    CommandResult() : wall_ms(0.0), exit_code(0) {}
    CommandResult(double w, int e, const std::string &o) : wall_ms(w), exit_code(e), output(o) {}
};

bool file_exists(const std::string &path) {
    return access(path.c_str(), F_OK) == 0;
}

std::string resolve_absolute_path(const std::string &path) {
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved) != nullptr) {
        return std::string(resolved);
    }
    return path;
}

std::vector<std::string> glob_files(const std::string &pattern) {
    glob_t results;
    std::vector<std::string> files;
    if (glob(pattern.c_str(), 0, nullptr, &results) == 0) {
        for (size_t i = 0; i < results.gl_pathc; ++i) {
            files.emplace_back(results.gl_pathv[i]);
        }
        std::sort(files.begin(), files.end());
    }
    globfree(&results);
    return files;
}

// Copy a file byte-for-byte from src to dst. Avoids a full JPEG decode+encode
// round-trip when we just need to archive an already-written debug image.
bool copy_file(const std::string &src, const std::string &dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::ofstream out(dst, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out << in.rdbuf();
    return out.good();
}

cv::Mat letterbox(const cv::Mat &image, int target_width, int target_height, cv::Scalar pad_color) {
    if (image.empty()) {
        return cv::Mat();
    }

    double scale = std::min(
        static_cast<double>(target_width) / static_cast<double>(image.cols),
        static_cast<double>(target_height) / static_cast<double>(image.rows));

    int resized_width = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    int resized_height = std::max(1, static_cast<int>(std::round(image.rows * scale)));

    cv::Mat resized;
    int interpolation = scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0, 0, interpolation);

    cv::Mat boxed(target_height, target_width, image.type(), pad_color);
    int pad_x = (target_width - resized_width) / 2;
    int pad_y = (target_height - resized_height) / 2;
    resized.copyTo(boxed(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    return boxed;
}

bool save_letterboxed_image(const std::string &input_path,
                            const std::string &output_path,
                            int width,
                            int height) {
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        return false;
    }

    cv::Mat boxed = letterbox(image, width, height, cv::Scalar(114, 114, 114));
    if (boxed.empty()) {
        return false;
    }

    return cv::imwrite(output_path, boxed);
}

CommandResult run_command_capture(const std::string &command) {
    auto start = std::chrono::high_resolution_clock::now();
    std::string shell_command = command + " 2>&1";
    FILE *pipe = popen(shell_command.c_str(), "r");
    if (pipe == nullptr) {
        return {0.0, -1, ""};
    }

    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
        std::cout << buffer;
    }

    int status = pclose(pipe);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    int exit_code = status;
    if (status != -1 && WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    }

    if (exit_code != 0) {
        std::cout << "Command failed (exit " << exit_code << "): " << command << std::endl;
    }

    return {elapsed.count(), exit_code, output};
}

// Extract a numeric value after the colon on a line containing `label`.
double parse_stage_value(const std::string &line, const std::string &label) {
    if (line.find(label) == std::string::npos) {
        return -1.0;
    }

    std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
        return -1.0;
    }

    std::size_t start = line.find_first_of("0123456789", colon);
    if (start == std::string::npos) {
        return -1.0;
    }

    std::size_t end = start;
    while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '.')) {
        ++end;
    }

    return std::stod(line.substr(start, end - start));
}

// Single-pass parser: scan the output once and extract all three timing values.
StageTiming parse_stage_timings(const std::string &text) {
    StageTiming timing;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        double val;
        val = parse_stage_value(line, "Pre-process Time");
        if (val >= 0.0) { timing.preprocess_ms = val; continue; }
        val = parse_stage_value(line, "NPU Inference Time");
        if (val >= 0.0) { timing.inference_ms = val; continue; }
        val = parse_stage_value(line, "CPU Post-Process Time");
        if (val >= 0.0) { timing.postprocess_ms = val; continue; }
    }
    return timing;
}

StageTiming parse_ocr_timing_file(const std::string &result_file) {
    StageTiming timing;
    std::ifstream input(result_file);
    if (!input.is_open()) {
        return timing;
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("preprocess_ms=", 0) == 0) {
            timing.preprocess_ms = std::stod(line.substr(std::strlen("preprocess_ms=")));
        } else if (line.rfind("inference_ms=", 0) == 0) {
            timing.inference_ms = std::stod(line.substr(std::strlen("inference_ms=")));
        } else if (line.rfind("postprocess_ms=", 0) == 0) {
            timing.postprocess_ms = std::stod(line.substr(std::strlen("postprocess_ms=")));
        }
    }

    return timing;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }

    const auto pipeline_start = std::chrono::high_resolution_clock::now();

    const std::string input_image = resolve_absolute_path(argv[1]);
    if (!file_exists(input_image)) {
        std::cerr << "Input image not found: " << argv[1] << std::endl;
        return 1;
    }

    ensure_output_dirs();

    // Clean up stale crops from previous runs so we don't accidentally read them
    for (const auto &f : glob_files(kMeterDetectDir + "/debug/meter_crop_*.jpg")) {
        unlink(f.c_str());
    }
    for (const auto &f : glob_files(kMeterCropDir + "/debug/text_crop_*.jpg")) {
        unlink(f.c_str());
    }

    const std::string pipeline_input_640 = kPipelineDebugDir + "/meter_detect_input_640x640.jpg";
    const std::string pipeline_meter_crop_640 = kPipelineDebugDir + "/meter_crop_input_640x640.jpg";

    auto stage_start = std::chrono::high_resolution_clock::now();
    if (!save_letterboxed_image(input_image, pipeline_input_640, 640, 640)) {
        std::cerr << "Failed to create 640x640 detector preview" << std::endl;
        return 1;
    }
    auto stage_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> detector_pre_ms = stage_end - stage_start;

    std::string detector_cmd = "cd " + shell_quote(kMeterDetectBuildDir) +
                               " && ./yolov8n -m ../data/best_int8.adla -p " + shell_quote(input_image);
    CommandResult detector_result = run_command_capture(detector_cmd);
    StageTiming detector_timing = parse_stage_timings(detector_result.output);

    std::vector<std::string> meter_crops = glob_files(kMeterDetectDir + "/debug/meter_crop_*.jpg");
    if (meter_crops.empty()) {
        std::cerr << "No meter crop was produced by meter_detect" << std::endl;
        return 1;
    }

    std::cout << "Meter detect produced " << meter_crops.size() << " crop(s)." << std::endl;

    const std::string selected_meter_crop = meter_crops.front();
    stage_start = std::chrono::high_resolution_clock::now();
    if (!save_letterboxed_image(selected_meter_crop, pipeline_meter_crop_640, 640, 640)) {
        std::cerr << "Failed to create 640x640 crop preview" << std::endl;
        return 1;
    }
    stage_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> crop_letterbox_ms = stage_end - stage_start;

    std::string crop_cmd = "cd " + shell_quote(kMeterCropBuildDir) +
                           " && ./yolov8n -m ../data/best_int8.adla -p " + shell_quote(pipeline_meter_crop_640);
    CommandResult crop_result = run_command_capture(crop_cmd);
    StageTiming crop_timing = parse_stage_timings(crop_result.output);

    std::vector<std::string> text_crops = glob_files(kMeterCropDir + "/debug/text_crop_*.jpg");
    if (text_crops.empty()) {
        std::cerr << "No text crop was produced by meter_crop" << std::endl;
        return 1;
    }

    std::cout << "Meter crop produced " << text_crops.size() << " text crop(s). OCR will run on all of them." << std::endl;

    std::ostringstream final_text_stream;
    std::vector<std::string> crop_readings;
    StageTiming ocr_totals;
    for (size_t i = 0; i < text_crops.size(); ++i) {
        std::cout << "OCR crop " << (i + 1) << " / " << text_crops.size() << ": " << text_crops[i] << std::endl;

        std::ostringstream processed_name;
        processed_name << kPipelineDebugDir << "/ocr_preprocessed_" << i << "_320x48.jpg";
        stage_start = std::chrono::high_resolution_clock::now();
        if (!save_letterboxed_image(text_crops[i], processed_name.str(), 320, 48)) {
            std::cerr << "Failed to save OCR input for: " << text_crops[i] << std::endl;
            continue;
        }
        stage_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ocr_letterbox_ms = stage_end - stage_start;

        std::string ocr_cmd = "cd " + shell_quote(kOcrDir) + " && /data/summerint26/venv/bin/python3 x.py " + shell_quote(processed_name.str());
        CommandResult ocr_result = run_command_capture(ocr_cmd);

        std::string result_file = kOcrDir + "/results/ocr_result.txt";
        std::string crop_reading;
        StageTiming ocr_timing = parse_ocr_timing_file(result_file);
        if (file_exists(result_file)) {
            std::ifstream result_stream(result_file);
            std::string line;
            if (std::getline(result_stream, line) && !line.empty()) {
                // Keep only the first 2 digits (last digit is insignificant)
                if (line.size() > 2) {
                    line = line.substr(0, 2);
                }
                crop_reading = line;
                final_text_stream << line;
            }
        }

        crop_readings.push_back(crop_reading);

        ocr_totals.preprocess_ms += ocr_letterbox_ms.count() + ocr_timing.preprocess_ms;
        ocr_totals.inference_ms += ocr_timing.inference_ms;
        ocr_totals.postprocess_ms += ocr_timing.postprocess_ms;

        std::ofstream per_crop_result(kPipelineResultsDir + "/ocr_crop_" + std::to_string(i) + ".txt");
        if (per_crop_result.is_open()) {
            per_crop_result << crop_reading << std::endl;
        }

        std::cout << "OCR crop " << (i + 1) << " wall time: " << ocr_result.wall_ms << " ms" << std::endl;
    }

    // Archive debug images to results dir via byte copy (no decode/encode round-trip).
    copy_file(pipeline_input_640, kPipelineResultsDir + "/detector_input_640x640.jpg");
    copy_file(pipeline_meter_crop_640, kPipelineResultsDir + "/meter_crop_input_640x640.jpg");

    std::ofstream pipeline_result(kPipelineResultsDir + "/final_reading.txt");
    if (pipeline_result.is_open()) {
        for (size_t i = 0; i < crop_readings.size(); ++i) {
            pipeline_result << "OCR crop " << (i + 1) << ": "
                            << (crop_readings[i].empty() ? "<empty>" : crop_readings[i])
                            << '\n';
        }
        pipeline_result << "Combined reading: " << final_text_stream.str() << '\n';
    }

    std::cout << "\n================ OCR SUMMARY ================\n";
    std::cout << "Text crops processed     : " << text_crops.size() << "\n";
    for (size_t i = 0; i < crop_readings.size(); ++i) {
        std::cout << "OCR crop " << (i + 1) << " reading : "
                  << (crop_readings[i].empty() ? "<empty>" : crop_readings[i]) << "\n";
    }
    std::cout << "OCR output assembly      : stored separately per crop\n";
    std::cout << "Per-crop OCR files       : " << kPipelineResultsDir << "/ocr_crop_*.txt\n";
    std::cout << "==================================================\n";

    const auto pipeline_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> total_pipeline_ms = pipeline_end - pipeline_start;

    const double total_preprocess_ms = detector_pre_ms.count() + detector_timing.preprocess_ms +
                                       crop_letterbox_ms.count() + crop_timing.preprocess_ms +
                                       ocr_totals.preprocess_ms;
    const double total_inference_ms = detector_timing.inference_ms + crop_timing.inference_ms + ocr_totals.inference_ms;
    const double total_postprocess_ms = detector_timing.postprocess_ms + crop_timing.postprocess_ms + ocr_totals.postprocess_ms;
    const double total_stage_ms = total_preprocess_ms + total_inference_ms + total_postprocess_ms;

    std::cout << "\n================ FINAL TIMING SUMMARY ================\n";
    std::cout << "PRE-PROCESS TIME  : " << total_preprocess_ms << " ms\n";
    std::cout << "NPU INFERENCE TIME: " << total_inference_ms << " ms\n";
    std::cout << "POST-PROCESS TIME : " << total_postprocess_ms << " ms\n";
    std::cout << "TOTAL TIME        : " << total_stage_ms << " ms\n";
    std::cout << "======================================================\n";

    std::ofstream total_time_report(kPipelineResultsDir + "/total_pipeline_time.txt");
    if (total_time_report.is_open()) {
        total_time_report << "PRE-PROCESS TIME  : " << total_preprocess_ms << " ms\n";
        total_time_report << "NPU INFERENCE TIME: " << total_inference_ms << " ms\n";
        total_time_report << "POST-PROCESS TIME : " << total_postprocess_ms << " ms\n";
        total_time_report << "TOTAL TIME        : " << total_stage_ms << " ms\n";
    }

    std::cout << "Results written to: " << kPipelineResultsDir << std::endl;
    return 0;
}