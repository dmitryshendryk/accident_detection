#include "opt_flow.hpp"
#include <zmq.hpp>


namespace pre_process 
{
OpticalFlow::OpticalFlow() {}
OpticalFlow::~OpticalFlow() {}

void OpticalFlow::convertFlowToImage(const Mat &flowIn, Mat &flowOut,
                                     float lowerBound, float higherBound)
{
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255 * ((v) - (L)) / ((H) - (L))))
    for (int i = 0; i < flowIn.rows; ++i)
    {
        for (int j = 0; j < flowIn.cols; ++j)
        {
            float x = flowIn.at<float>(i, j);
            flowOut.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
        }
    }
#undef CAST
}


int OpticalFlow::compute_Flow(int start_with_vid, int gpuID, int type, int frameSkip,
                              std::string vid_path, std::string out_path, std::string out_path_jpeg)
{
    float MIN_SZ = 256;
    float OUT_SZ = 256;

    bool clipFlow = true; // clips flow to [-20 20]
    bool resize_img = false;

    // These are default paths

    bool createOutDirs = true;

    /* THESE ARE MY PARAMS, NOT FEICHENHOFER'S */

    bool debug = true;
    bool rgb = false;
    bool bins = false;

    // Global variables for cuda::BroxOpticalFlow
    const float alpha_ = 0.197;
    const float gamma_ = 50;
    const float scale_factor_ = 0.8;
    const int inner_iterations_ = 10;
    const int outer_iterations_ = 77;
    const int solver_iterations_ = 10;

    const int RESIZE_WIDTH = 224;
    const int RESIZE_HEIGHT = 224;
    const bool warp = false;

    int vidcount = 0;

    int totalvideos = 0;
    struct dirent *entry;

    cv::cuda::setDevice(gpuID);
    Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray, human_mask;

    // cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    /////// zmq client 

    //////////////

    Ptr<cuda::BroxOpticalFlow> dflow = cuda::BroxOpticalFlow::create(alpha_, gamma_, scale_factor_, inner_iterations_, outer_iterations_, solver_iterations_);

    Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();

    int vidID = 0;
    std::string video, outfile_u, outfile_v, outfile_flow, outfile_jpeg;

        // cout << vid_path << endl;

        VideoCapture cap;
        try
        {  
            cap.open("/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/videos/YoutubeVid2.mp4");
            
        }
        catch (std::exception &e)
        {
            
            // std::cout << e.what() << '\n';
        }

        int nframes = 0, width = 0, height = 0, width_out = 0, height_out = 0;
        float factor = 0, factor_out = 0;

        if (cap.isOpened() == 0)
        {
            return -1;
        }

        cap >> frame1_rgb_;
        if (resize_img == true)
        {
            factor = std::max<float>(MIN_SZ / frame1_rgb_.cols, MIN_SZ / frame1_rgb_.rows);

            width = std::floor(frame1_rgb_.cols * factor);
            width -= width % 2;
            height = std::floor(frame1_rgb_.rows * factor);
            height -= height % 2;

            frame1_rgb = cv::Mat(Size(width, height), CV_8UC3);
            width = frame1_rgb.cols;
            height = frame1_rgb.rows;
            cv::resize(frame1_rgb_, frame1_rgb, cv::Size(224, 224), 0, 0, INTER_CUBIC);

            factor_out = std::max<float>(OUT_SZ / width, OUT_SZ / height);

            rgb_out = cv::Mat(Size(cvRound(width * factor_out), cvRound(height * factor_out)), CV_8UC3);
            width_out = rgb_out.cols;
            height_out = rgb_out.rows;
        }
        else
        {
            frame1_rgb = cv::Mat(Size(frame1_rgb_.cols, frame1_rgb_.rows), CV_8UC3);
            width = frame1_rgb.cols;
            height = frame1_rgb.rows;
            frame1_rgb_.copyTo(frame1_rgb);
        }

        // Allocate memory for the images
        frame0_rgb = cv::Mat(Size(width, height), CV_8UC3);
        flow_rgb = cv::Mat(Size(width, height), CV_8UC3);
        motion_flow = cv::Mat(Size(width, height), CV_8UC3);
        frame0 = cv::Mat(Size(width, height), CV_8UC1);
        frame1 = cv::Mat(Size(width, height), CV_8UC1);
        frame0_32 = cv::Mat(Size(width, height), CV_32FC1);
        frame1_32 = cv::Mat(Size(width, height), CV_32FC1);

        // Convert the image to grey and float
        cvtColor(frame1_rgb, frame1, CV_BGR2GRAY);
        frame1.convertTo(frame1_32, CV_32FC1, 1.0 / 255.0, 0);

        // outfile_u = out_folder_u.toStdString();
        // outfile_v = out_folder_v.toStdString();

        QString outfile_u = QString::fromStdString(out_path + "u/");
        QString outfile_v = QString::fromStdString(out_path + "v/");

        // outfile_u = "/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/output_u/";
        // outfile_v = "/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/output_v/";
        // outfile_flow  = out_folder_flow.toStdString();
        // cout << frame1.empty()  << endl;
        while (frame1.empty() == false)
        {
            gettimeofday(&tod1, NULL);
            t1fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
            if (nframes >= 1)
            {
                gettimeofday(&tod1, NULL);
                //	GetSystemTime(&tod1);
                t1 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
                switch (type)
                {
                case 0:
                    frame1GPU.upload(frame1_32);
                    frame0GPU.upload(frame0_32);
                    dflow->calc(frame0GPU, frame1GPU, flowGPU);
                case 1:
                    frame1GPU.upload(frame1);
                    frame0GPU.upload(frame0);
                    alg_tvl1->calc(frame0GPU, frame1GPU, flowGPU);
                }

                flowGPU.download(flowCPU);
                cv::split(flowCPU, planes);
                imgU = planes[0];
                imgV = planes[1];

                gettimeofday(&tod1, NULL);
                t2 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
                tdflow = 1000.0 * (t2 - t1);
            }

            if (WRITEOUT_IMGS == true && nframes >= 1)
            {
                if (resize_img == true)
                {

                    cv::resize(imgU, imgU, cv::Size(224, 224), 0, 0, INTER_CUBIC);
                    cv::resize(imgV, imgV, cv::Size(224, 224), 0, 0, INTER_CUBIC);
                }

                double min_u, max_u;
                cv::minMaxLoc(imgU, &min_u, &max_u);
                double min_v, max_v;
                cv::minMaxLoc(imgV, &min_v, &max_v);

                float min_u_f = min_u;
                float max_u_f = max_u;

                float min_v_f = min_v;
                float max_v_f = max_v;

                if (clipFlow)
                {
                    min_u_f = -20;
                    max_u_f = 20;

                    min_v_f = -20;
                    max_v_f = 20;
                }

                cv::Mat img_u(imgU.rows, imgU.cols, CV_8UC1);
                cv::Mat img_v(imgV.rows, imgV.cols, CV_8UC1);

                Mat img_u_cal, img_v_cal, mag_nor;

                mag_nor = imgU;

                convertFlowToImage(imgU, img_u, min_u_f, max_u_f);
                convertFlowToImage(imgV, img_v, min_v_f, max_v_f);

                pow(imgU, 2.0f, planes[0]);
                pow(imgV, 2.0f, planes[1]);
                add(planes[0], planes[1], planes[2], noArray(), CV_32F);
                sqrt(planes[2], mag);
                convertFlowToImage(mag, mag_nor, 0, 20 * 1.414f);

                Mat optflow(imgU.rows, imgU.cols, CV_8UC3);
                for (int i = 0; i < img_u.rows; ++i)
                {
                    for (int j = 0; j < img_u.cols; ++j)
                    {

                        optflow.at<Vec3b>(i, j)[0] = img_u.at<uchar>(i, j);
                        optflow.at<Vec3b>(i, j)[1] = img_v.at<uchar>(i, j);
                        optflow.at<Vec3b>(i, j)[2] = mag_nor.at<uchar>(i, j);
                    }
                }

                sprintf(cad, "/frame%06d.jpg", nframes);

                imwrite(outfile_u + cad, img_u);
                imwrite(outfile_v + cad, img_v);

                // ct1->pImg_V = img_v;
                // ct1->pImg_U = img_u;
                // *pImg_V = img_v;
                
                // std::cout << pImg_V->rows << std::endl;
               
 

                // cout << img_u.rows << endl;
                // cout << img_u.channels() << endl;

                
                //imwrite(outfile_flow+cad, optflow);

                if (bins == true)
                {
                    //  fwrite(&min_u_f,sizeof(float),1,fx);
                    // fwrite(&max_u_f,sizeof(float),1,fx);
                    // fwrite(&min_v_f,sizeof(float),1,fx);
                    // fwrite(&max_v_f,sizeof(float),1,fx);
                }
            }

            sprintf(cad, "/my_frames_frame%06d.jpg", nframes + 1);
            if (rgb)
            {
                if (resize_img == true)
                {
                    cv::resize(frame1_rgb, rgb_out, cv::Size(224, 224), 0, 0, INTER_CUBIC);
                    imwrite(outfile_jpeg + cad, rgb_out);
                }
                else
                    imwrite(outfile_jpeg + cad, frame1_rgb);
            }
            if (debug)
            {
                
            }
            frame1_rgb.copyTo(frame0_rgb);
            cvtColor(frame0_rgb, frame0, CV_BGR2GRAY);
            frame0.convertTo(frame0_32, CV_32FC1, 1.0 / 255.0, 0);

            nframes++;
            for (int iskip = 0; iskip < frameSkip; iskip++)
            {
                cap >> frame1_rgb_;
            }
            if (frame1_rgb_.empty() == false)
            {
                if (resize_img == true)
                {
                    cv::resize(frame1_rgb_, frame1_rgb, cv::Size(224, 224), 0, 0, INTER_CUBIC);
                }
                else
                {
                    frame1_rgb_.copyTo(frame1_rgb);
                }

                cvtColor(frame1_rgb, frame1, CV_BGR2GRAY);
                frame1.convertTo(frame1_32, CV_32FC1, 1.0 / 255.0, 0);
            }
            else
            {
                break;
            }

            gettimeofday(&tod1, NULL);
            if (debug)
            {
                t2fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
                tdframe = 1000.0 * (t2fr - t1fr);
            }
        }
        if (bins == true)
        {
            //            fclose(fx);
        }
    // }
}
}