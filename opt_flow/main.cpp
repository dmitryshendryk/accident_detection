#include<iostream>
#include "opt_flow.hpp"
using namespace std;
int main(int argc, char *argv[] )
{

    const char* keys = 
				"{ help h usage ? |       | print this message   }"
				"{ start_video    |   1   | start video id       }"
				"{ gpuID          |   1   | use this gpu         }"
				"{ type           |   1   | use this flow method (0=Brox, 1=TV-L1)}"
				"{ skip           |   1   | frame skip           }"
				"{ vid_path       |  ./   | path input (where the videos are)}"
				"{ out_path       |  ./   | path output          }";

    int start_with_vid = 1;
    int gpuID = 0;
    int type = 1;
    int frameSkip = 1;

    std::string vid_path = "/home/katou2/github-home/gpu_flow/build/test";
    std::string out_path	= "/home/katou2/github-home/gpu_flow/build/test_out";
    std::string out_path_jpeg	= "/home/katou2/github-home/gpu_flow/build/test_out";

    
    CommandLineParser parser(argc, argv, keys);

	if (parser.get<bool>("help"))
	{
		cout << "Usage: compute_flow [options]" << endl;
		cout << "Avaible options:" << endl;
		parser.printMessage();
		return 0;
	}

	if (argc > 1) {
		start_with_vid = parser.get<int>("start_video");
		gpuID = parser.get<int>("gpuID");
		type = parser.get<int>("type");
		frameSkip = parser.get<int>("skip");
		vid_path = parser.get<std::string>("vid_path");
		out_path = parser.get<std::string>("out_path");
		out_path_jpeg = out_path + "/rgb/";
		cout << "start_vid:" << start_with_vid << "gpuID:" << gpuID << "flow method: "<< type << " frameSkip: " << frameSkip << " vid_path: " << vid_path << " out_path" << out_path << " jpegs: " << out_path_jpeg << endl;
	}

    pre_process::OpticalFlow opt_flow;
    opt_flow.compute_Flow(int start_with_vid, int gpuID, int type, int frameSkip,
                        String vid_path, String out_path, String out_path_jpeg);
    return 0;
}