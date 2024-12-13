#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <mutex>
#include <atomic>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <future>
#include <optional>

const std::string ASCII_CHARS = ".-:=+*%#@$";
std::mutex progress_mutex;
std::atomic<int> completed_frames(0);

// Função para mapear valor do pixel para caractere ASCII
char pixel_to_ascii(double pixel) {
    return ASCII_CHARS[static_cast<int>(pixel * ASCII_CHARS.size() / 256)];
}

// Aplica cor ao caractere ASCII baseado nos valores RGB
std::string apply_color_to_ascii(const cv::Vec3b &pixel) {
    int r = pixel[2], g = pixel[1], b = pixel[0];
    return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m"
           + pixel_to_ascii((r + g + b) / 3.0);
}

// Converte uma frame de imagem para uma string ASCII com cores
std::string convert_frame_to_ascii(const cv::Mat &frame) {
    std::ostringstream ascii_frame;
    for (int i = 0; i < frame.rows; ++i) {
        for (int j = 0; j < frame.cols; ++j) {
            ascii_frame << apply_color_to_ascii(frame.at<cv::Vec3b>(i, j));
        }
        ascii_frame << '\n';
    }
    return ascii_frame.str();
}

// Exibe a barra de progresso com atualizações esporádicas para evitar lock excessivo
void display_progress(float progress) {
    constexpr int bar_width = 50;
    static float last_progress = -1.0f;

    if (progress - last_progress < 0.05f) return; // Atualiza a cada 5%
    last_progress = progress;

    std::lock_guard<std::mutex> lock(progress_mutex);
    std::cout << "[";
    int pos = static_cast<int>(bar_width * progress);
    for (int i = 0; i < bar_width; ++i) {
        std::cout << (i < pos ? "=" : (i == pos ? ">" : " "));
    }
    std::cout << "] " << static_cast<int>(progress * 100) << " %\r";
    std::cout.flush();
}

// Processa os frames em threads separadas usando `std::async`
void process_frames(int start, int end, const std::vector<cv::Mat> &frames, std::vector<std::string> &ascii_frames,
                    int total_frames) {
    for (int i = start; i < end; ++i) {
        ascii_frames[i] = convert_frame_to_ascii(frames[i]);
        ++completed_frames;
        display_progress(static_cast<float>(completed_frames) / total_frames);
    }
}

// Função para processar argumentos
std::optional<std::string> parse_arguments(int argc, char *argv[], bool &preload, bool &clear_screen) {
    std::string video_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--preload" || arg == "-p") preload = true;
        else if (arg == "--clear" || arg == "-c") clear_screen = true;
        else if ((arg == "--video" || arg == "-v") && i + 1 < argc) video_path = argv[++i];
    }
    return video_path.empty() ? std::nullopt : std::optional<std::string>(video_path);
}

// Exibe o vídeo em ASCII
void display_ascii_video(const std::vector<std::string> &ascii_frames, bool clear_screen, float frame_rate) {
    for (const auto &frame: ascii_frames) {
        if (clear_screen) std::cout << "\033[H\033[2J";
        else std::cout << "\033[H";
        std::cout << frame;
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0f / frame_rate)));
    }
}

// Função principal
int main(int argc, char *argv[]) {
    bool preload = false, clear_screen = false;
    auto video_path = parse_arguments(argc, argv, preload, clear_screen);
    if (!video_path) {
        std::cerr << "Por favor, forneça o caminho do vídeo com a flag --video ou -v." << std::endl;
        return -1;
    }

    cv::VideoCapture video_capture(*video_path);
    if (!video_capture.isOpened()) {
        std::cerr << "Erro: não foi possível abrir o vídeo." << std::endl;
        return -1;
    }

    // Parâmetros do vídeo
    const float frame_rate = video_capture.get(cv::CAP_PROP_FPS);
    const int frame_width = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
    const int frame_height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int custom_width = 480;
    const int custom_height = static_cast<int>((custom_width * frame_height / frame_width) * 0.4);
    const int total_frames = video_capture.get(cv::CAP_PROP_FRAME_COUNT);

    std::vector<cv::Mat> frames;
    cv::Mat frame, resized_frame;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (preload) {
        // Carrega todos os frames antes de converter para ASCII
        while (video_capture.read(frame)) {
            cv::resize(frame, resized_frame, cv::Size(custom_width, custom_height), 0, 0, cv::INTER_AREA);
            frames.emplace_back(resized_frame.clone());
        }

        // Converte frames para ASCII usando threads assíncronas
        std::vector<std::string> ascii_frames(frames.size());
        int num_threads = std::thread::hardware_concurrency();
        int frames_per_thread = frames.size() / num_threads;

        std::vector<std::future<void> > futures;
        for (int i = 0; i < num_threads; ++i) {
            int start = i * frames_per_thread;
            int end = (i == num_threads - 1) ? frames.size() : (i + 1) * frames_per_thread;
            futures.emplace_back(std::async(std::launch::async, process_frames, start, end, std::cref(frames),
                                            std::ref(ascii_frames), total_frames));
        }
        for (auto &fut: futures) fut.get();

        std::cout << std::endl;
        display_ascii_video(ascii_frames, clear_screen, frame_rate);
    } else {
        // Converte e exibe frame a frame em tempo real
        while (video_capture.read(frame)) {
            cv::resize(frame, resized_frame, cv::Size(custom_width, custom_height), 0, 0, cv::INTER_AREA);
            std::string ascii_frame = convert_frame_to_ascii(resized_frame);
            if (clear_screen) std::cout << "\033[H\033[2J";
            else std::cout << "\033[H";
            std::cout << ascii_frame;
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0f / frame_rate)));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\033[0m";
    std::cout << "Framerate: " << frame_rate << " FPS\n"
            << "Height: " << custom_height << "\n"
            << "Width: " << custom_width << "\n"
            << "Execution Time: " << duration.count() << " ms" << std::endl;

    return 0;
}
