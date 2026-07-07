/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Falldown AI inference single-header interface.
 */

#ifndef AI_INFERENCE_H
#define AI_INFERENCE_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nncase/functional/ai2d/ai2d_builder.h>
#include <nncase/runtime/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;
using namespace nncase::F::k230;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;

struct face_coordinate
{
    int x1;
    int y1;
    int x2;
    int y2;
};

struct landmarks_t
{
    float points[10];
};

typedef struct
{
    std::vector<face_coordinate> boxes;
    std::vector<landmarks_t> landmarks;
} DetectResult;

class falldownDetect;

struct AiContext
{
    falldownDetect *model;
    size_t channel;
    size_t height;
    size_t width;
};

int ai_init(AiContext *ctx,
            const char *kmodel_path,
            size_t channel,
            size_t height,
            size_t width);

DetectResult ai_run_frame(AiContext *ctx,
                          uintptr_t vaddr,
                          uintptr_t paddr);

void ai_deinit(AiContext *ctx);

class ScopedTiming
{
public:
	/**
	 * @brief ScopedTiming鏋勯€犲嚱鏁?鍒濆鍖栬鏃跺璞″悕绉板苟寮€濮嬭鏃?	 * @param info 			 璁℃椂瀵硅薄鍚嶇О
	 * @param enable_profile 鏄惁寮€濮嬭鏃?	 * @return None
	 */
	ScopedTiming(std::string info = "ScopedTiming", int enable_profile = 1)
		: m_info(info), enable_profile(enable_profile)
	{
		if (enable_profile)
		{
			m_start = std::chrono::steady_clock::now();
		}
	}

	/**
	 * @brief ScopedTiming鏋愭瀯,缁撴潫璁℃椂锛屽苟鎵撳嵃鑰楁椂
	 * @return None
	 */
	~ScopedTiming()
	{
		if (enable_profile)
		{
			m_stop = std::chrono::steady_clock::now();
			double elapsed_ms = std::chrono::duration<double, std::milli>(m_stop - m_start).count();
			std::cout << m_info << " took " << elapsed_ms << " ms" << std::endl;
		}
	}

private:
	int enable_profile;							   // 鏄惁缁熻鏃堕棿
	std::string m_info;							   // 璁℃椂瀵硅薄鍚嶇О
	std::chrono::steady_clock::time_point m_start; // timing start
	std::chrono::steady_clock::time_point m_stop;  // timing stop
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

/**
 * @brief 浜鸿劯妫€娴嬫
 */
typedef struct Bbox
{
    float x; // 浜鸿劯妫€娴嬫鐨勫乏椤剁偣x鍧愭爣
    float y; // 浜鸿劯妫€娴嬫鐨勫乏椤剁偣x鍧愭爣
    float w;
    float h;
} Bbox;

/**
 * @brief 浜鸿劯浜斿畼鐐? */
typedef struct SparseLandmarks
{
    float points[10];
} SparseLandmarks;

/**
 * @brief 鍗曞紶/甯у浘鐗囧ぇ灏? */
typedef struct FrameSize
{
    size_t width;
    size_t height;
} FrameSize;

/**
 * @brief 鍗曞紶/甯у浘鐗囧ぇ灏? */
typedef struct FrameCHWSize
{
    size_t channel;
    size_t height;
    size_t width;
} FrameCHWSize;

/**
 * @brief AI Demo宸ュ叿绫? * 灏佽浜咥I Demo甯哥敤鐨勫嚱鏁帮紝鍖呮嫭浜岃繘鍒舵枃浠惰鍙栥€佹枃浠朵繚瀛樸€佸浘鐗囬澶勭悊绛夋搷浣? */
class Utils
{
public:
    /**
     * @brief 璇诲彇2杩涘埗鏂囦欢
     * @param file_name 鏂囦欢璺緞
     * @return 鏂囦欢瀵瑰簲绫诲瀷鐨勬暟鎹?     */
    template <class T>
    static vector<T> read_binary_file(const char *file_name)
    {
        ifstream ifs(file_name, std::ios::binary);
        ifs.seekg(0, ifs.end);
        size_t len = ifs.tellg();
        vector<T> vec(len / sizeof(T), 0);
        ifs.seekg(0, ifs.beg);
        ifs.read(reinterpret_cast<char *>(vec.data()), len);
        ifs.close();
        return vec;
    }

    /**
     * @brief 鎵撳嵃鏁版嵁
     * @param data 闇€鎵撳嵃鏁版嵁瀵瑰簲鎸囬拡
     * @param size 闇€鎵撳嵃鏁版嵁澶у皬
     * @return None
     */
    template <class T>
    static void dump(const T *data, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            cout << data[i] << " ";
        }
        cout << endl;
    }

    // 闈欐€佹垚鍛樺嚱鏁颁笉渚濊禆浜庣被鐨勫疄渚嬶紝鍙互鐩存帴閫氳繃绫诲悕璋冪敤
    /**
     * @brief 灏嗘暟鎹互2杩涘埗鏂瑰紡淇濆瓨涓烘枃浠?     * @param file_name 淇濆瓨鏂囦欢璺緞+鏂囦欢鍚?     * @param data      闇€瑕佷繚瀛樼殑鏁版嵁
     * @param size      闇€瑕佷繚瀛樼殑闀垮害
     * @return None
     */
    static void dump_binary_file(const char *file_name, char *data, const size_t size);

    /**
     * @brief 灏嗘暟鎹繚瀛樹负鐏板害鍥剧墖
     * @param file_name  淇濆瓨鍥剧墖璺緞+鏂囦欢鍚?     * @param frame_size 淇濆瓨鍥剧墖鐨勫銆侀珮
     * @param data       闇€瑕佷繚瀛樼殑鏁版嵁
     * @return None
     */
    static void dump_gray_image(const char *file_name, const FrameSize &frame_size, unsigned char *data);

    /**
     * @brief 灏嗘暟鎹繚瀛樹负褰╄壊鍥剧墖
     * @param file_name  淇濆瓨鍥剧墖璺緞+鏂囦欢鍚?     * @param frame_size 淇濆瓨鍥剧墖鐨勫銆侀珮
     * @param data       闇€瑕佷繚瀛樼殑鏁版嵁
     * @return None
     */
    static void dump_color_image(const char *file_name, const FrameSize &frame_size, unsigned char *data);


    /*************************for img process********************/
    /**
     * @brief 瀵瑰浘鐗囪繘琛屽厛padding鍚巖esize鐨勫鐞?     * @param ori_img             鍘熷鍥剧墖
     * @param frame_size      闇€瑕乺esize鍥惧儚鐨勫楂?     * @param padding         闇€瑕乸adding鐨勫儚绱狅紝榛樿鏄痗v::Scalar(114, 114, 114),BGR
     * @return 澶勭悊鍚庡浘鍍?     */
    static cv::Mat padding_resize(const cv::Mat img, const FrameSize &frame_size, const cv::Scalar &padding = cv::Scalar(114, 114, 114));

    /**
     * @brief 瀵瑰浘鐗噐esize
     * @param ori_img             鍘熷鍥剧墖
     * @param frame_size      闇€瑕乺esize鍥惧儚鐨勫楂?     * @param padding         闇€瑕乸adding鐨勫儚绱狅紝榛樿鏄痗v::Scalar(114, 114, 114),BGR
     * @return                澶勭悊鍚庡浘鍍?     */
    static cv::Mat resize(const cv::Mat ori_img, const FrameSize &frame_size);

    /**
     * @brief 灏嗗浘鐗囦粠bgr杞负rgb
     * @param ori_img         鍘熷鍥剧墖
     * @return                澶勭悊鍚庡浘鍍?     */
    static cv::Mat bgr_to_rgb(cv::Mat ori_img);

    /**
     * @brief 灏哛GB鎴朢GB鍥剧墖浠巋wc杞负chw
     * @param ori_img          鍘熷鍥剧墖
     * @param chw_vec          杞负chw鍚庣殑鏁版嵁
     * @return None
     */
    static void hwc_to_chw(cv::Mat &ori_img, std::vector<uint8_t> &chw_vec); // for rgb data

    /**
     * @brief 灏咮GR鍥剧墖浠巋wc杞负chw
     * @param ori_img          鍘熷鍥剧墖
     * @param chw_vec          杞负chw鍚庣殑鏁版嵁
     * @return None
     */
    static void bgr2rgb_and_hwc2chw(cv::Mat &ori_img, std::vector<uint8_t> &chw_vec);

    /*************************for ai2d ori_img process********************/
    // resize
    /**
     * @brief resize鍑芥暟锛屽chw鏁版嵁杩涜resize
     * @param ori_shape        鍘熷鏁版嵁chw
     * @param chw_vec          鍘熷鏁版嵁
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @return None
     */
    void resize(FrameCHWSize ori_shape, std::vector<uint8_t> &chw_vec, runtime_tensor &ai2d_out_tensor);

    /**
     * @brief resize鍑芥暟
     * @param builder          ai2d鏋勫缓鍣紝鐢ㄤ簬杩愯ai2d
     * @param ai2d_in_tensor   ai2d杈撳叆
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @return None
     */
    void resize(std::unique_ptr<ai2d_builder> &builder, runtime_tensor &ai2d_in_tensor, runtime_tensor &ai2d_out_tensor);

    // crop resize
    /**
     * @brief resize鍑芥暟锛屽chw鏁版嵁杩涜crop & resize
     * @param builder          ai2d鏋勫缓鍣紝鐢ㄤ簬杩愯ai2d
     * @param ai2d_in_tensor   ai2d杈撳叆
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @return None
     */
    void crop_resize(FrameCHWSize ori_shape, std::vector<uint8_t> &chw_vec, Bbox &crop_info, runtime_tensor &ai2d_out_tensor);
    
    /**
     * @brief crop_resize鍑芥暟锛屽chw鏁版嵁杩涜crop & resize
     * @param crop_info        闇€瑕乧rop鐨勪綅缃紝x,y,w,h
     * @param builder          ai2d鏋勫缓鍣紝鐢ㄤ簬杩愯ai2d
     * @param ai2d_in_tensor   ai2d杈撳叆
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @return None
     */
    void crop_resize(Bbox &crop_info, std::unique_ptr<ai2d_builder> &builder, runtime_tensor &ai2d_in_tensor, runtime_tensor &ai2d_out_tensor);

    // padding resize
    /**
     * @brief padding_resize鍑芥暟锛堜笂涓嬪乏鍙硃adding锛夛紝瀵筩hw鏁版嵁杩涜padding & resize
     * @param ori_shape        鍘熷鏁版嵁chw
     * @param chw_vec          鍘熷鏁版嵁
     * @param builder          ai2d鏋勫缓鍣紝鐢ㄤ簬杩愯ai2d
     * @param ai2d_in_tensor   ai2d杈撳叆
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @return None
     */
    static void padding_resize(FrameCHWSize ori_shape, std::vector<uint8_t> &chw_vec, FrameSize resize_shape, runtime_tensor &ai2d_out_tensor, cv::Scalar padding);
    
    /**
     * @brief padding_resize鍑芥暟锛堝彸鎴栦笅padding锛夛紝瀵筩hw鏁版嵁杩涜padding & resize
     * @param ori_shape        鍘熷鏁版嵁chw
     * @param chw_vec          鍘熷鏁版嵁
     * @param resize_shape     resize涔嬪悗鐨勫ぇ灏?     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @param padding          濉厖鍊硷紝鐢ㄤ簬resize鏃剁殑绛夋瘮渚嬪彉鎹?     * @return None
     */
    static void padding_resize_one_side(FrameCHWSize ori_shape, std::vector<uint8_t> &chw_vec, FrameSize resize_shape, runtime_tensor &ai2d_out_tensor, cv::Scalar padding);

    /**
     * @brief padding_resize鍑芥暟锛堜笂涓嬪乏鍙硃adding锛夛紝瀵筩hw鏁版嵁杩涜padding & resize
     * @param ori_shape        鍘熷鏁版嵁chw
     * @param resize_shape     resize涔嬪悗鐨勫ぇ灏?     * @param builder          ai2d鏋勫缓鍣紝鐢ㄤ簬杩愯ai2d
     * @param ai2d_in_tensor   ai2d杈撳叆
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @param padding          濉厖鍊硷紝鐢ㄤ簬resize鏃剁殑绛夋瘮渚嬪彉鎹?     * @return None
     */
    static void padding_resize(FrameCHWSize ori_shape, FrameSize resize_shape, std::unique_ptr<ai2d_builder> &builder, runtime_tensor &ai2d_in_tensor, runtime_tensor &ai2d_out_tensor, cv::Scalar padding);
    
    /**
     * @brief padding_resize鍑芥暟锛堝彸鎴栦笅padding锛夛紝瀵筩hw鏁版嵁杩涜padding & resize
     * @param ori_shape        鍘熷鏁版嵁chw
     * @param resize_shape     resize涔嬪悗鐨勫ぇ灏?     * @param builder          ai2d鏋勫缓鍣紝鐢ㄤ簬杩愯ai2d
     * @param ai2d_in_tensor   ai2d杈撳叆
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @param padding          濉厖鍊硷紝鐢ㄤ簬resize鏃剁殑绛夋瘮渚嬪彉鎹?     * @return None
     */
    static void padding_resize_one_side(FrameCHWSize ori_shape, FrameSize resize_shape, std::unique_ptr<ai2d_builder> &builder, runtime_tensor &ai2d_in_tensor, runtime_tensor &ai2d_out_tensor, const cv::Scalar padding);

    // affine
    /**
     * @brief 浠垮皠鍙樻崲鍑芥暟锛屽chw鏁版嵁杩涜浠垮皠鍙樻崲(for imgae)
     * @param ori_shape        鍘熷鏁版嵁chw澶у皬
     * @param ori_data         鍘熷鏁版嵁
     * @param affine_matrix    浠垮皠鍙樻崲鐭╅樀
     * @param ai2d_out_tensor  浠垮皠鍙樻崲鍚庣殑鏁版嵁
     * @return None
     */
    static void affine(FrameCHWSize ori_shape, std::vector<uint8_t> &ori_data, float *affine_matrix, runtime_tensor &ai2d_out_tensor);

    /**
     * @brief 浠垮皠鍙樻崲鍑芥暟锛屽chw鏁版嵁杩涜浠垮皠鍙樻崲(for video)
     * @param affine_matrix    浠垮皠鍙樻崲鐭╅樀
     * @param builder          ai2d鏋勫缓鍣紝鐢ㄤ簬杩愯ai2d
     * @param ai2d_in_tensor   ai2d杈撳叆
     * @param ai2d_out_tensor  ai2d杈撳嚭
     * @return None
     */
    static void affine(float *affine_matrix, std::unique_ptr<ai2d_builder> &builder, runtime_tensor &ai2d_in_tensor, runtime_tensor &ai2d_out_tensor);

    /**
     * @brief 闈炴瀬澶у€兼姂鍒?     * @input_boxes 鎵€鏈夊€欓€夋
     * @NMS_THRESH NMS闃堝€?
     * @return None
    **/
    static void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

    /**
     * @brief 瑙ｇ爜
     * @data kmodel 鎺ㄧ悊缁撴灉
     * @net_size kmodel 杈撳叆灏哄澶у皬
     * @stride 姝ラ暱
     * @num_classes 绫诲埆鏁?     * @frame_size 鍒嗚鲸鐜?     * @anchors 閿氭
     * @threshold 妫€娴嬫闃堝€?    **/
    static std::vector<BoxInfo> decode_infer(float *data, int net_size, int stride, int num_classes, FrameSize frame_size, float anchors[][2], float threshold);
};

class AIBase
{
public:
    /**
     * @brief AI鍩虹被鏋勯€犲嚱鏁帮紝鍔犺浇kmodel,骞跺垵濮嬪寲kmodel杈撳叆銆佽緭鍑?     * @param kmodel_file kmodel鏂囦欢璺緞
     * @param debug_mode  0锛堜笉璋冭瘯锛夈€?1锛堝彧鏄剧ず鏃堕棿锛夈€?锛堟樉绀烘墍鏈夋墦鍗颁俊鎭級
     * @return None
     */
    AIBase(const char *kmodel_file,const string model_name, const int debug_mode = 1);

    /**
     * @brief AI鍩虹被鏋愭瀯鍑芥暟
     * @return None
     */
    ~AIBase();

    /**
     * @brief 鏍规嵁绱㈠紩鑾峰彇kmodel杈撳叆tensor
     * @param idx 杈撳叆鏁版嵁鎸囬拡
     * @return None
     */
    runtime_tensor get_input_tensor(size_t idx);

    /**
     * @brief 鎺ㄧ悊kmodel
     * @return None
     */
    void run();

    /**
     * @brief 鑾峰彇kmodel杈撳嚭锛岀粨鏋滀繚瀛樺湪瀵瑰簲鐨勭被灞炴€т腑
     * @return None
     */
    void get_output();

protected:
    string model_name_;
    int debug_mode_;
    vector<float *> p_outputs_;
    vector<vector<float>> output_buffers_;
    vector<vector<int>> input_shapes_;
    vector<vector<int>> output_shapes_;
    vector<int> each_input_size_by_byte_;
    vector<int> each_output_size_by_byte_;
private:
    /**
     * @brief 棣栨鍒濆鍖杒model杈撳叆锛屽苟鑾峰彇杈撳叆shape
     * @return None
     */
    void set_input_init();

    /**
     * @brief 棣栨鍒濆鍖杒model杈撳嚭锛屽苟鑾峰彇杈撳嚭shape
     * @return None
     */
    void set_output_init();

    interpreter kmodel_interp_;        // kmodel瑙ｉ噴鍣紝浠巏model鏂囦欢鏋勫缓锛岃礋璐ｆā鍨嬬殑鍔犺浇銆佽緭鍏ヨ緭鍑鸿缃拰鎺ㄧ悊
    vector<unsigned char> kmodel_vec_; // 閫氳繃璇诲彇kmodel鏂囦欢寰楀埌鏁翠釜kmodel鏁版嵁锛岀敤浜庝紶缁檏model瑙ｉ噴鍣ㄥ姞杞絢model
};

class falldownDetect: public AIBase
{
    public:

        /** 
        * for image
        * @brief falldownDetect 鏋勯€犲嚱鏁帮紝鍔犺浇kmodel,骞跺垵濮嬪寲kmodel杈撳叆銆佽緭鍑恒€佺被闃堝€煎拰NMS闃堝€?        * @param kmodel_file kmodel鏂囦欢璺緞
        * @param obj_thresh 妫€娴嬫闃堝€?        * @param nms_thresh NMS闃堝€?        * @param debug_mode 0锛堜笉璋冭瘯锛夈€?1锛堝彧鏄剧ず鏃堕棿锛夈€?锛堟樉绀烘墍鏈夋墦鍗颁俊鎭級
        * @return None
        */
        falldownDetect(const char *kmodel_file, float obj_thresh,float nms_thresh,  const int debug_mode);

        /** 
        * for video
        * @brief falldownDetect 鏋勯€犲嚱鏁帮紝鍔犺浇kmodel,骞跺垵濮嬪寲kmodel杈撳叆銆佽緭鍑恒€佺被闃堝€煎拰NMS闃堝€?        * @param kmodel_file kmodel鏂囦欢璺緞
        * @param obj_thresh 妫€娴嬫闃堝€?        * @param nms_thresh NMS闃堝€?        * @param isp_shape   isp杈撳叆澶у皬锛坈hw锛?        * @param vaddr       isp瀵瑰簲铏氭嫙鍦板潃
        * @param paddr       isp瀵瑰簲鐗╃悊鍦板潃
        * @param debug_mode 0锛堜笉璋冭瘯锛夈€?1锛堝彧鏄剧ず鏃堕棿锛夈€?锛堟樉绀烘墍鏈夋墦鍗颁俊鎭級
        * @return None
        */
        falldownDetect(const char *kmodel_file, float obj_thresh,float nms_thresh, FrameCHWSize isp_shape, uintptr_t vaddr, uintptr_t paddr, const int debug_mode);
        /** 
        * @brief  falldownDetect 鏋愭瀯鍑芥暟
        * @return None
        */
        ~falldownDetect();

        /**
         * @brief 鍥剧墖棰勫鐞嗭紙ai2d for image锛?         * @param ori_img 鍘熷鍥剧墖
         * @return None
         */
        void pre_process(cv::Mat ori_img);

        /**
         * @brief 瑙嗛娴侀澶勭悊锛坅i2d for video锛?         * @return None
         */
        void pre_process();

        /**
         * @brief kmodel鎺ㄧ悊
         * @return None
         */
        void inference();

        /** 
        * @brief postprocess 鍑芥暟锛屽杈撳嚭瑙ｇ爜鍚庣殑缁撴灉锛岃繘琛孨MS澶勭悊
        * @param frame_size 甯уぇ灏?        * @param result   鎵€鏈夊€欓€夋娴嬫
        * @return None
        */
        void post_process(FrameSize frame_size,std::vector<BoxInfo> &result);

        void set_frame(uintptr_t vaddr, uintptr_t paddr);

        std::vector<std::string> labels { "Normal", "Fall" }; // 绫诲埆鏍囩

    private:
        float obj_thresh_;  // object threshold
        float nms_thresh_;  // NMS闃堝€?        
        int anchors_num_ = 3;  // 閿氭涓暟
        int classes_num_ = 2;   // class count
        int channels_ = anchors_num_ * (5 + classes_num_);
        float anchors_0_[3][2] = { { 10, 13 }, { 16, 30 }, { 33, 23 } };
        float anchors_1_[3][2] = { { 30, 61 }, { 62, 45 }, { 59, 119 } };
        float anchors_2_[3][2] = { { 116, 90 }, { 156, 198 }, { 373, 326 } };
        std::unique_ptr<ai2d_builder> ai2d_builder_; // ai2d builder
        runtime_tensor ai2d_in_tensor_;              // ai2d杈撳叆tensor
        runtime_tensor ai2d_out_tensor_;             // ai2d杈撳嚭tensor
        uintptr_t vaddr_;                            // isp鐨勮櫄鎷熷湴鍧€
        FrameCHWSize isp_shape_;                     // isp瀵瑰簲鐨勫湴鍧€澶у皬

};

#endif
