// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    The face detector we use is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
       One Millisecond Face Alignment with an Ensemble of Regression Trees by
       Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset (see
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
       C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
       300 faces In-the-wild challenge: Database and results. 
       Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
    You can get the trained model file from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
    Note that the license for the iBUG 300-W dataset excludes commercial use.
    So you should contact Imperial College London to find out if it's OK for
    you to use this model file in a commercial product.


    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#pragma comment(lib, "wsock32.lib")
#include <winsock2.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
//#include <dlib/image_io.h>
#include <opencv2\opencv.hpp>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
	const int maxPartsNum = 68;

	// We need a face detector.  We will use this to get bounding boxes for
	// each face in an image.
	frontal_face_detector detector = get_frontal_face_detector();
	// And we also need a shape_predictor.  This is the tool that will predict face
	// landmark positions given an image and face bounding box.  Here we are just
	// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
	// as a command line argument.
	shape_predictor sp;
	deserialize("C:\\dlib\\examples\\build\\shape_predictor_68_face_landmarks.dat") >> sp;

	cv::VideoCapture cap(0);//デバイスのオープン
							//cap.open(0);//こっちでも良い．

	if (!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
	{
		//読み込みに失敗したときの処理
		return -1;
	}

	// ウィンドウの準備
	cv::namedWindow("window");
	cv::moveWindow("window", 0, 0);

	// ====================== UDP通信の準備
	char* address = "127.0.0.1";	// IPアドレス
	unsigned short port = 50001;	// ポート番号
	ULONG* buffer = new ULONG[maxPartsNum * 2 + 1];	// 座標格納用buffer
	//winsockの初期化
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1, 1), &wsadata);
	HOSTENT* pHostent = gethostbyname(address);
	if (pHostent == NULL) {
		DWORD addr = inet_addr(address);
		pHostent = gethostbyaddr((char*)&addr, 4, AF_INET);
	}
	if (pHostent == NULL) return -1;//サーバーが見つかりません
	SOCKET s = socket(AF_INET, SOCK_DGRAM, 0);
	if (s<0) return -2;//ソケットエラー
	SOCKADDR_IN sockaddrin;
	memset(&sockaddrin, 0, sizeof(sockaddrin));
	memcpy(&(sockaddrin.sin_addr), pHostent->h_addr_list[0], pHostent->h_length);
	sockaddrin.sin_family = AF_INET;
	sockaddrin.sin_port = htons(port);

	int resizeRate = 2; // light ver: 3
	while (1)//無限ループ
	{
		cv::TickMeter meter;
		meter.start();

		cv::Mat frame, frameBGR, frameResized;
		cap >> frame; // get a new frame from camera

		cv::flip(frame, frame, 1);
		cv::resize(frame, frame, cv::Size(frame.cols / resizeRate, frame.rows / resizeRate));

		// 黒目の位置検出
		int eyeLXmin, eyeLXmax, eyeLYmin, eyeLYmax;
		int eyeRXmin, eyeRXmax, eyeRYmin, eyeRYmax;
		//std::cout << "frame.type(): " << frame.type() << " CV_8UC3:" << CV_8UC3 << std::endl;
		cv::Mat eyeLImg(cv::Size(1, 1), CV_8UC1);
		cv::Mat eyeRImg(cv::Size(1, 1), CV_8UC1);

		// 旧
		//cv::cvtColor(frame, frameBGR, cv::COLOR_BGR2RGB);
		//array2d<rgb_pixel> img;
		////OpenCV→dlib変換
		//assign_image(img, cv_image<bgr_pixel>(frameBGR));

		//OpenCV→dlib変換
		cv_image<bgr_pixel> img(frame);

		//// Make the image larger so we can detect small faces.
		//pyramid_up(img);	// これを入れると遅くなる

		// Now tell the face detector to give us a list of bounding boxes
		// around all the faces in the image.
		std::vector<rectangle> dets = detector(img);
		//cout << "Number of faces detected: " << dets.size() << endl;

		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector<full_object_detection> shapes;
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
			full_object_detection shape = sp(img, dets[j]);
			//cout << "number of parts: " << shape.num_parts() << endl;
			//cout << "pixel position of first part:  " << shape.part(0) << endl;
			//cout << "pixel position of second part: " << shape.part(1) << endl;
			// You get the idea, you can get all the face part locations if
			// you want them.  Here we just store them in shapes so we can
			// put them on the screen.
			shapes.push_back(shape);


			// 黒目検出用
			eyeLYmax = 0; eyeRYmax = 0;
			eyeLYmin = frame.rows; eyeRYmin = frame.rows;
			// eyeL
			eyeLXmax = shapes[0].part(45).x();
			eyeLXmin = shapes[0].part(42).x();
			eyeLYmax = shapes[0].part(46).y() > shapes[0].part(47).y() ? shapes[0].part(46).y() : shapes[0].part(47).y();
			eyeLYmin = shapes[0].part(43).y() < shapes[0].part(44).y() ? shapes[0].part(43).y() : shapes[0].part(44).y();
			// eyeR
			eyeRXmax = shapes[0].part(39).x();
			eyeRXmin = shapes[0].part(36).x();
			eyeRYmax = shapes[0].part(40).y() > shapes[0].part(41).y() ? shapes[0].part(40).y() : shapes[0].part(41).y();
			eyeRYmin = shapes[0].part(37).y() < shapes[0].part(38).y() ? shapes[0].part(37).y() : shapes[0].part(38).y();
			// 目の画像を白黒に変換しつつコピー
			cv::cvtColor(cv::Mat(frame, cv::Rect(eyeLXmin, eyeLYmin, eyeLXmax - eyeLXmin, eyeLYmax - eyeLYmin)), eyeLImg, cv::COLOR_BGR2GRAY);
			cv::cvtColor(cv::Mat(frame, cv::Rect(eyeRXmin, eyeRYmin, eyeRXmax - eyeRXmin, eyeRYmax - eyeRYmin)), eyeRImg, cv::COLOR_BGR2GRAY);

//			for (int j = 0; j < shape.num_parts(); j++) {
			for (int i = 0; i < maxPartsNum; i++) {
				// pyramid_upすると、座標の値がぴったり1/2になる
				cv::circle(frame, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255));
				buffer[2 * i + 0] = shape.part(i).x()*resizeRate;
				buffer[2 * i + 1] = shape.part(i).y()*resizeRate;
			}
		}

		// 黒目検出
		//std::cout << "eyeLXmin, eyeLXmax, eyeLYmin, eyeLYmax: "
		//	<< eyeLXmin << " " << eyeLXmax << " " << eyeLYmin << " " << eyeLYmax << std::endl;
		//std::cout << "eyeLImg.type(): " << eyeLImg.type() << " CV_8UC1:" << CV_8UC1 << std::endl;
		eyeLImg *= 1.2;
		eyeRImg *= 1.2;
		cv::Mat eyeLImgThres120;
		cv::threshold(eyeLImg, eyeLImgThres120, 120, 255, cv::THRESH_BINARY_INV);
		cv::Mat eyeRImgThres120;
		cv::threshold(eyeRImg, eyeRImgThres120, 120, 255, cv::THRESH_BINARY_INV);
		//cv::imshow("eyeLImgThres120", eyeLImgThres120);
		//cv::imshow("eyeRImgThres120", eyeRImgThres120);

		cv::Moments moment;
		cv::Point2f eyeLsight(0,0), eyeRsight(0, 0);
		float eyeLCenterRate, eyeRCenterRate, eyeAveCenterRate;
		moment = cv::moments(eyeLImgThres120, false);
		if(moment.m00)
			eyeLsight = cv::Point2f(moment.m10 / moment.m00, moment.m01 / moment.m00);
		moment = cv::moments(eyeRImgThres120, false);
		if (moment.m00)
			eyeRsight = cv::Point2f(moment.m10 / moment.m00, moment.m01 / moment.m00);
		//cv::circle(eyeLImg, eyeLsight, 4, cv::Scalar(255, 0, 0), 1, 4);
		//cv::circle(eyeRImg, eyeRsight, 4, cv::Scalar(255, 0, 0), 1, 4);
		//cv::imshow("eyeLimg", eyeLImg);
		//cv::imshow("eyeRimg", eyeRImg);
		eyeLCenterRate = eyeLsight.x / (float)eyeLImg.cols;
		eyeRCenterRate = eyeRsight.x / (float)eyeRImg.cols;
		eyeAveCenterRate = (eyeLCenterRate + eyeRCenterRate) / 2.0f;
		if (eyeAveCenterRate) {
		//	std::cout << "eyeLCenterRate, eyeRCenterRate, eyeAveCenterRate: "
		//		<< eyeLCenterRate << " " << eyeRCenterRate << " " << eyeAveCenterRate << std::endl;
			cv::Point2i eyeSight((shapes[0].part(42).x() - shapes[0].part(39).x())*eyeAveCenterRate + shapes[0].part(39).x(), (shapes[0].part(42).y() - shapes[0].part(39).y())*eyeAveCenterRate + shapes[0].part(39).y());
			cv::circle(frame, eyeSight, 2, cv::Scalar(255, 0, 0), 3);
		}
		// 黒目検出結果をバッファに入れる
		buffer[2 * maxPartsNum + 0] = *(ULONG*)&eyeAveCenterRate;

		// ====================== UDP送信
		int re = sendto(s, (char*)buffer, (2 * maxPartsNum+1) * sizeof(ULONG), 0, (LPSOCKADDR)&sockaddrin, sizeof(sockaddrin));
		//std::cout << "bytes: " << 2 * maxPartsNum * sizeof(ULONG) << std::endl;


		meter.stop();
		std::cout << meter.getTimeMilli() << " ms" << std::endl;

		cv::imshow("window", frame);//画像を表示．

		int key = cv::waitKey(1);
		if (key == 113)//qボタンが押されたとき
		{
			break;//whileループから抜ける．
		}
		else if (key == '1')// large size : slow
		{
			resizeRate = 1;
		}
		else if (key == '2')// midium size : medium
		{
			resizeRate = 2;
		}
		else if (key == '3')// small size : fast
		{
			resizeRate = 3;
		}
		else if (key == 115)//sが押されたとき
		{
			//フレーム画像を保存する．
			cv::imwrite("img.png", frame);
		}
	}
	// メモリの開放
	delete[] buffer;

	// ====================== UDP通信の後処理
	closesocket(s);
	//winsockのクリーンアップ
	WSACleanup();

	cv::destroyAllWindows();

	return 0;

}

