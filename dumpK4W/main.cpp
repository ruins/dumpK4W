// REAL WORLD TESTING
// 1) COLLECT DATASET WITH EXTENSION CABLE
// 2) 15FPS Color - More light?

// GENERAL TODOs
// 1) READ UP on COM model

#include <Kinect.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>
#include <mutex>

#include <algorithm>

// VS2012 (VC11) doesn't have C++11 std round...
namespace std
{
    int round(double d)
    {
        return static_cast<int>(d + 0.5);
    }
}

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
using std::thread;
using std::mutex;
using std::stringstream;
using std::ofstream;
using std::round;

// TODO Do we need this?
// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
    if (pInterfaceToRelease != NULL)
    {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}

// Same for Depth and Infrared
static const Size DEPTH_SIZE = Size(512, 424);
static const int DEPTH_DEPTH = 2;
static const int DEPTH_PIXEL_TYPE = CV_16UC1;
static const int DEPTH_MAGIC_NUMBER = 18;	// Scales depth up to allow OpenCV visualisation

// Note that Raw color is YUY2 (Flipped UYVY)
static const Size COLOR_SIZE = Size(1920, 1080);
static const int COLOR_DEPTH = 2;
//static const int COLOR_PIXEL_TYPE = CV_8UC2;

static const char* DUMP_PATH = "E:/dump/";

// Relative Time is in 100ns "Ticks". Divide by these to get us or ms
static const INT64 TICKS_TO_US = 10;
static const INT64 TICKS_TO_MS = 10000;

static const INT32 NUM_SECONDS_TO_CAPTURE = 30; 
static const INT32 MAX_FRAMES_TO_CAPTURE = 30 * NUM_SECONDS_TO_CAPTURE;	


// ---- Globals for the sake of convenience :) ----
// Kinect v2 stuff
static IKinectSensor* kinect = NULL;
static IDepthFrameReader *depthReader = NULL;
static IInfraredFrameReader *infraReader = NULL;
static IColorFrameReader *colorReader = NULL;
static ICoordinateMapper *coordMapper = NULL;

// Frame Data buffers
static Mat **depthImageArray = NULL;
static UINT16 *depthBufArray[MAX_FRAMES_TO_CAPTURE];
static Mat **infraImageArray = NULL;
static UINT16 *infraBufArray[MAX_FRAMES_TO_CAPTURE];
static BYTE *colorBufArray[MAX_FRAMES_TO_CAPTURE];

// Time Stamps (relative)
static TIMESPAN depthRelTimeArray[MAX_FRAMES_TO_CAPTURE];
static TIMESPAN infraRelTimeArray[MAX_FRAMES_TO_CAPTURE];
static TIMESPAN colorRelTimeArray[MAX_FRAMES_TO_CAPTURE];

// Number of frames captured to RAM
static int DEPTH_FRAMES_CAPTURED = 0;
static int INFRA_FRAMES_CAPTURED = 0;
static int COLOR_FRAMES_CAPTURED = 0;

// Signals
static bool CAPTURE_DONE = false;	// Signal used by all threads. True => break loop

// Threading
std::mutex ioMutex;


void ProcessDepth()
{
	HRESULT hr;

	// Depth
	IDepthFrameSource *depthSrc = NULL;
	hr = kinect->get_DepthFrameSource(&depthSrc);
	if(FAILED(hr)) exit(EXIT_FAILURE);

	hr = depthSrc->OpenReader(&depthReader);
	if(FAILED(hr)) exit(EXIT_FAILURE);
	SafeRelease(depthSrc);

	// Subscribing reader
	WAITABLE_HANDLE depthHandle = 0;
	hr = depthReader->SubscribeFrameArrived(&depthHandle);
	if(FAILED(hr)) exit(EXIT_FAILURE);

	// Depth Data and Visualization Init
	depthImageArray = new Mat*[MAX_FRAMES_TO_CAPTURE];
	memset(depthImageArray, 0, sizeof(Mat*)*MAX_FRAMES_TO_CAPTURE);
	memset(depthBufArray, 0, sizeof(UINT16*)*MAX_FRAMES_TO_CAPTURE);
	memset(depthRelTimeArray, 0, sizeof(TIMESPAN)*MAX_FRAMES_TO_CAPTURE);
	for(int i = 0; i < MAX_FRAMES_TO_CAPTURE; ++i)
		depthBufArray[i] = new UINT16[DEPTH_SIZE.area()];
	namedWindow("Depth", WINDOW_AUTOSIZE);

	int i;
	for(i = 0; i < MAX_FRAMES_TO_CAPTURE && !CAPTURE_DONE;)
	{
		DWORD ret = WaitForSingleObject((HANDLE)depthHandle, 200) ;

		if(ret == WAIT_TIMEOUT) {
			std::cerr << "!!!Depth Timeout!!!" << endl;
			std::cerr << i << endl;
		}
		else if (ret != WAIT_OBJECT_0) {
			std::cerr << "!!!Depth Error!!!" << endl;
			if(ret == WAIT_FAILED)
				std::cerr << GetLastError() << endl;
		}
		else {
			if(!depthReader)
				exit(EXIT_FAILURE);

			IDepthFrameArrivedEventArgs* pArgs = nullptr;
			depthReader->GetFrameArrivedEventData(depthHandle, &pArgs);

			IDepthFrameReference *depthRef = nullptr;
			pArgs->get_FrameReference(&depthRef);

			//hr = depthReader->AcquireLatestFrame(&depthFrame);
			IDepthFrame* depthFrame = NULL;

			if(SUCCEEDED(depthRef->AcquireFrame(&depthFrame))) 
			{
				// Copying data from Kinect
				depthFrame->CopyFrameDataToArray(DEPTH_SIZE.area(), depthBufArray[i]);

				// Saving timestamp
				depthFrame->get_RelativeTime(depthRelTimeArray + i);

				depthFrame->Release();
		
				depthImageArray[i] = new Mat(DEPTH_SIZE, DEPTH_PIXEL_TYPE, depthBufArray[i], Mat::AUTO_STEP);

				imshow("Depth", *depthImageArray[i] * DEPTH_MAGIC_NUMBER);

				++i;	// Incrementing frame number
			}

			pArgs->Release();
		}

		if(waitKey(1) == 'q') {
			break;
		}
	}
	DEPTH_FRAMES_CAPTURED = i;
	ioMutex.lock();
		cout << "ProcessDepth Thread DONE!!" << endl;
		cout << "Depth frames in RAM: " << DEPTH_FRAMES_CAPTURED<< endl;
	ioMutex.unlock();

	CAPTURE_DONE = true;
}

void ProcessInfra()
{
	CAPTURE_DONE = false;	// We are not done yet!

	HRESULT hr;

	// Infrared
	IInfraredFrameSource *infraSrc = NULL;
	hr = kinect->get_InfraredFrameSource(&infraSrc);
	if(FAILED(hr)) exit(EXIT_FAILURE);

	hr = infraSrc->OpenReader(&infraReader);
	if(FAILED(hr)) exit(EXIT_FAILURE);
	SafeRelease(infraSrc);

	// Subscribing reader
	WAITABLE_HANDLE infraHandle = 0;
	infraReader->SubscribeFrameArrived(&infraHandle);
	if(FAILED(hr)) exit(EXIT_FAILURE);
	

	// Infrared
	infraImageArray = new Mat*[MAX_FRAMES_TO_CAPTURE];
	memset(infraImageArray, 0, sizeof(Mat*)*MAX_FRAMES_TO_CAPTURE);
	memset(infraBufArray, 0, sizeof(UINT16*)*MAX_FRAMES_TO_CAPTURE);
	memset(infraRelTimeArray, 0, sizeof(TIMESPAN)*MAX_FRAMES_TO_CAPTURE);
	for(int i = 0; i < MAX_FRAMES_TO_CAPTURE; ++i)
		infraBufArray[i] = new UINT16[DEPTH_SIZE.area()];
	namedWindow("Infra", WINDOW_AUTOSIZE);	

	int i;
	for(i = 0; i < MAX_FRAMES_TO_CAPTURE && !CAPTURE_DONE;)
	{
		DWORD ret = WaitForSingleObject((HANDLE)infraHandle, 200) ;

		if(ret == WAIT_TIMEOUT) {
			std::cerr << "!!!Infra Timeout!!!" << endl;
		}
		else if (ret != WAIT_OBJECT_0) {
			std::cerr << "!!!Infra Error!!!" << endl;
			if(ret == WAIT_FAILED)
				std::cerr << GetLastError() << endl;
		}
		else {
			if(!infraReader)
				exit(EXIT_FAILURE);

			IInfraredFrameArrivedEventArgs* pArgs = nullptr;
			infraReader->GetFrameArrivedEventData(infraHandle, &pArgs);

			IInfraredFrameReference *infraRef = nullptr;
			pArgs->get_FrameReference(&infraRef);

			//hr = depthReader->AcquireLatestFrame(&depthFrame);
			IInfraredFrame* infraFrame = NULL;

			if(SUCCEEDED(infraRef->AcquireFrame(&infraFrame))) 
			{
				// Copying data from Kinect
				infraFrame->CopyFrameDataToArray(DEPTH_SIZE.area(), infraBufArray[i]);

				// Saving timestamp
				infraFrame->get_RelativeTime(infraRelTimeArray + i);

				infraFrame->Release();
		
				infraImageArray[i] = new Mat(DEPTH_SIZE, DEPTH_PIXEL_TYPE, infraBufArray[i], Mat::AUTO_STEP);
				imshow("Infra", *infraImageArray[i]);

				++i;	// Incrementing frame number
			}

			pArgs->Release();
		}

		if(waitKey(1) == 'q') {
			break;
		}
	}	
	INFRA_FRAMES_CAPTURED = i;
	ioMutex.lock();
		cout << "ProcessInfra Thread DONE!!" << endl;
		cout << "Infra frames in RAM: " << INFRA_FRAMES_CAPTURED << endl;
	ioMutex.unlock();

	CAPTURE_DONE = true;
}

void ProcessColor()
{
	HRESULT hr;

	// Color		
	IColorFrameSource *colorSrc = NULL;
	hr = kinect->get_ColorFrameSource(&colorSrc);
	if(FAILED(hr)) exit(EXIT_FAILURE);

	hr = colorSrc->OpenReader(&colorReader);
	if(FAILED(hr)) exit(EXIT_FAILURE);
	SafeRelease(colorSrc);

	// Subscribing reader
	WAITABLE_HANDLE colorHandle = 0;
	colorReader->SubscribeFrameArrived(&colorHandle);
	if(FAILED(hr)) exit(EXIT_FAILURE);

	

	// Color
	//Mat **colorImageArray = new Mat*[MAX_FRAMES_TO_CAPTURE];
	//memset(colorImageArray, 0, sizeof(Mat*)*MAX_FRAMES_TO_CAPTURE);
	memset(colorBufArray, 0, sizeof(BYTE*)*MAX_FRAMES_TO_CAPTURE);
	memset(colorRelTimeArray, 0, sizeof(TIMESPAN)*MAX_FRAMES_TO_CAPTURE);
	for(int i = 0; i < MAX_FRAMES_TO_CAPTURE; ++i)
		colorBufArray[i] = new BYTE[COLOR_SIZE.area()*COLOR_DEPTH];
	//namedWindow("Color", WINDOW_AUTOSIZE);

	int i;
	for(i = 0; i < MAX_FRAMES_TO_CAPTURE && !CAPTURE_DONE;)
	{

		DWORD ret = WaitForSingleObject((HANDLE)colorHandle, 200) ;

		if(ret == WAIT_TIMEOUT) {
			std::cerr << "!!!Color Timeout!!!" << endl;
		}
		else if (ret != WAIT_OBJECT_0) {
			std::cerr << "!!!Color Error!!!" << endl;
			if(ret == WAIT_FAILED)
				std::cerr << GetLastError() << endl;
		}
		else {
			if(!colorReader)
				exit(EXIT_FAILURE);

			IColorFrameArrivedEventArgs* pArgs = nullptr;
			colorReader->GetFrameArrivedEventData(colorHandle, &pArgs);

			IColorFrameReference *colorRef = nullptr;
			pArgs->get_FrameReference(&colorRef);

			//hr = infraReader->AcquireLatestFrame(&infraFrame);
			IColorFrame* colorFrame = NULL;

			if(SUCCEEDED(colorRef->AcquireFrame(&colorFrame))) 
			{
				// Copying data from Kinect
				colorFrame->CopyRawFrameDataToArray(COLOR_SIZE.area()*COLOR_DEPTH, reinterpret_cast<BYTE*>(colorBufArray[i]));

				// Saving timestamp
				colorFrame->get_RelativeTime(colorRelTimeArray + i);

				colorFrame->Release();

				// TODO How to visualise RGB? Maybe just show Y channel (C1)
				//colorImageArray[cc] = new Mat(COLOR_SIZE, COLOR_PIXEL_TYPE, colorBufArray[cc], Mat::AUTO_STEP);
				//imshow("Color", *colorImageArray[cc]);

				++i;
			}

			pArgs->Release();
		}

		if(waitKey(1) == 'q') {
			break;
		}
	}

	// TODO do we need this if we have COLOR_FRAMES_CAPTURED?
	// We need this as all colorBufArray values are non-zero (aka valid)
	if(i < MAX_FRAMES_TO_CAPTURE) {
		delete [] colorBufArray[i];
		colorBufArray[i] = NULL;
	}

	COLOR_FRAMES_CAPTURED = i;

	ioMutex.lock();
		cout << "ProcessColor Thread DONE!!" << endl;
		cout << "Color Frames in RAM: " << COLOR_FRAMES_CAPTURED << endl;
	ioMutex.unlock();

	CAPTURE_DONE = true;
}

void WriteDepth()
{
	ofstream out(string(DUMP_PATH) + "depth_times.txt");
	if(out.bad()) {
		cerr << "Problem opening depth_times.txt" << endl;
		exit(EXIT_FAILURE);
	}
	out << "frame_idx" << "\t" << "RelativeTime" << endl;

	int i;
	for(i = 0; i < DEPTH_FRAMES_CAPTURED; ++i)
	{	
		// Not dumping extra frames without corresponding images at the end
		if(depthImageArray[i] == 0)
			break;

		// Generating numbered filename and dumping images to disk
		stringstream depthFilename;
		depthFilename << DUMP_PATH << "depth"; 
		depthFilename.width(8);
		depthFilename.fill('0');
		depthFilename << i;
		depthFilename << ".tiff";

//		ioMutex.lock();
			cout << "Writing: " << depthFilename.str() << endl;
			imwrite(depthFilename.str().c_str(), *depthImageArray[i]);

			// Timestamp
			out << i << "\t" << depthRelTimeArray[i] << endl;		
//		ioMutex.unlock();
	}
	ioMutex.lock();
		cout << "WriteDepth Thread DONE!!" << endl;
		cout << "Depth Frames written to " << DUMP_PATH << endl;
		cout << "Depth Frames written: " << i << endl;
	ioMutex.unlock();

}

void WriteInfra()
{
	ofstream out(string(DUMP_PATH) + "infra_times.txt");
	if(out.bad()) {
		cerr << "Problem opening infra_times.txt" << endl;
		exit(EXIT_FAILURE);
	}
	out << "frame_idx" << "\t" << "RelativeTime" << endl;

	int i;
	for(i = 0; i < INFRA_FRAMES_CAPTURED; ++i)
	{
		
		// Not dumping extra frames without corresponding images at the end
		if(infraImageArray[i] == 0)
			break;

		// Generating numbered filename and dumping images to disk
		stringstream infraFilename;
		infraFilename << DUMP_PATH << "infra"; 
		infraFilename.width(8);
		infraFilename.fill('0');
		infraFilename << i;
		infraFilename << ".tiff";

//		ioMutex.lock();
			cout << "Writing: " << infraFilename.str() << endl;
			imwrite(infraFilename.str().c_str(), *infraImageArray[i]);

			// Timestamp
			out << i << "\t" << infraRelTimeArray[i] << endl;
//		ioMutex.unlock();

	}
	ioMutex.lock();
		cout << "WriteInfra Thread DONE!!" << endl;
		cout << "Infra Frames written to " << DUMP_PATH << endl;
		cout << "Infra Frames written: " << i << endl;
	ioMutex.unlock();
}

void WriteColor()
{
	ofstream out(string(DUMP_PATH) + "color_times.txt");
	if(out.bad()) {
		cerr << "Problem opening color_times.txt" << endl;
		exit(EXIT_FAILURE);
	}
	out << "frame_idx" << "\t" << "RelativeTime" << endl;

	BYTE *grayBuf = new BYTE[COLOR_SIZE.area()];	// Y channel data of YUY2
	BYTE *rgbBuf = new BYTE[COLOR_SIZE.area() * 3];
	ColorSpacePoint *depthInColorSpace = new ColorSpacePoint[DEPTH_SIZE.area()];
	BYTE *grayBufMapped = new BYTE[DEPTH_SIZE.area()];
	BYTE *rgbBufMapped = new BYTE[DEPTH_SIZE.area()*3];

	int i;
	for(i = 0; i < COLOR_FRAMES_CAPTURED; ++i)
	{
		if(colorBufArray[i] == 0)
			break;

		// Dumping YUY2 raw color to files
		stringstream colorFilename;
		colorFilename << DUMP_PATH << "yuyv"; 
		colorFilename.width(8);
		colorFilename.fill('0');
		colorFilename << i;
		colorFilename << ".yuv";

//	ioMutex.lock();
		cout << "Writing: " << colorFilename.str() << endl;
		FILE* colorFile;
		colorFile = fopen(colorFilename.str().c_str(), "wb");
		fwrite(colorBufArray[i], COLOR_SIZE.area(), COLOR_DEPTH, colorFile);
		fclose(colorFile);
//	ioMutex.unlock();

		// Filling grayBuf with Y channel
		BYTE* cBuf = colorBufArray[i];
		for(int x = 0; x < COLOR_SIZE.area(); ++x)
		{
			grayBuf[x] = cBuf[2*x];
		}

		// Using OpenCV Mat header to wrap and save
		stringstream grayFilename;
		grayFilename << DUMP_PATH << "gray"; 
		grayFilename.width(8);
		grayFilename.fill('0');
		grayFilename << i;
		grayFilename << ".tiff";


		Mat gray(COLOR_SIZE, CV_8UC1, grayBuf, Mat::AUTO_STEP);


//		ioMutex.lock();		
		cout << "Writing: " << grayFilename.str() << endl;		
		imwrite(grayFilename.str().c_str(), gray);		
//		ioMutex.unlock();

		// YUY2 to RGB according to:
		// http://stackoverflow.com/questions/4491649/how-to-convert-yuy2-to-a-bitmap-in-c
		BYTE *ptrIn = colorBufArray[i];
		BYTE *ptrOut = rgbBuf;
		for (int j = 0;  j < COLOR_SIZE.area()/2;  ++j)
		{
			int y0 = ptrIn[0];
			int u0 = ptrIn[1];
			int y1 = ptrIn[2];
			int v0 = ptrIn[3];
			ptrIn += 4;
			int c = y0 - 16;
			int d = u0 - 128;
			int e = v0 - 128;
			ptrOut[0] = saturate_cast<uchar>(( 298 * c + 516 * d + 128) >> 8); // blue
			ptrOut[1] = saturate_cast<uchar>(( 298 * c - 100 * d - 208 * e + 128) >> 8); // green
			ptrOut[2] = saturate_cast<uchar>(( 298 * c + 409 * e + 128) >> 8); // red
			c = y1 - 16;
			ptrOut[3] = saturate_cast<uchar>(( 298 * c + 516 * d + 128) >> 8); // blue
			ptrOut[4] = saturate_cast<uchar>(( 298 * c - 100 * d - 208 * e + 128) >> 8); // green
			ptrOut[5] = saturate_cast<uchar>(( 298 * c + 409 * e + 128) >> 8); // red
			ptrOut += 6;
		}

		stringstream rgbFilename;
		rgbFilename << DUMP_PATH << "rgb"; 
		rgbFilename.width(8);
		rgbFilename.fill('0');
		rgbFilename << i;
		rgbFilename << ".tiff";

		Mat rgb(COLOR_SIZE, CV_8UC3, rgbBuf, Mat::AUTO_STEP);

		cout << "Writing: " << rgbFilename.str() << endl;		
		imwrite(rgbFilename.str().c_str(), rgb);	

		// REMAP TO DEPTH SPACE
		// TODO speed up with LUT?
		// TODO dump depth coords?
		// TODO can speed this up if we guess 15FPS etc

		// Finding nearest depthBuffer in terms of Relative Time
		int lastDepthIdx = DEPTH_FRAMES_CAPTURED-1;
		for(int j = 0; j < DEPTH_FRAMES_CAPTURED; ++j)
		{
			if(colorRelTimeArray[i] < depthRelTimeArray[j]) {
				lastDepthIdx = j-1;
				break;
			}
		}

		UINT16 *depthBuf = depthBufArray[lastDepthIdx];
		if(depthBuf) {
			//HRESULT hr = coordMapper->MapColorFrameToDepthSpace(DEPTH_SIZE.area(), depthBuf
			//	, COLOR_SIZE.area(), colorInDepthSpace);
			//if(FAILED(hr)) exit(EXIT_FAILURE);
			
			HRESULT hr = coordMapper->MapDepthFrameToColorSpace(DEPTH_SIZE.area(), depthBuf
				, DEPTH_SIZE.area(), depthInColorSpace);
			if(FAILED(hr)) {
				std::cerr << "COLOR MAPPING FAILED!!" << endl;
				std::cerr << (unsigned long)hr << endl;
				exit(EXIT_FAILURE);	
			}

			memset(grayBufMapped, 0, sizeof(BYTE)*DEPTH_SIZE.area());
			memset(rgbBufMapped, 0, sizeof(BYTE)*DEPTH_SIZE.area() * 3);
			for(int j = 0; j < DEPTH_SIZE.area(); ++j)
			{
				int x = round(depthInColorSpace[j].X);
				int y = round(depthInColorSpace[j].Y);

				if(x >= 0 && x < COLOR_SIZE.width && y >=0 && y < COLOR_SIZE.height) {
					grayBufMapped[j] = gray.at<BYTE>(y, x);
					
					// TODO speed up
					Vec3b bgrPixel = rgb.at<Vec3b>(y, x);
					rgbBufMapped[3*j] = bgrPixel[0];
					rgbBufMapped[3*j+1] = bgrPixel[1];
					rgbBufMapped[3*j+2] = bgrPixel[2];
				}
			}

			Mat grayMapped = Mat(DEPTH_SIZE, CV_8UC1, grayBufMapped, Mat::AUTO_STEP);
			Mat rgbMapped =  Mat(DEPTH_SIZE, CV_8UC3, rgbBufMapped, Mat::AUTO_STEP);

			stringstream grayMappedFilename;
			grayMappedFilename << DUMP_PATH << "grayMapped"; 
			grayMappedFilename.width(8);
			grayMappedFilename.fill('0');
			grayMappedFilename << i;
			grayMappedFilename << ".tiff";

			cout << "Writing: " << grayMappedFilename.str() << endl;		
			imwrite(grayMappedFilename.str().c_str(), grayMapped);

			stringstream rgbMappedFilename;
			rgbMappedFilename << DUMP_PATH << "rgbMapped"; 
			rgbMappedFilename.width(8);
			rgbMappedFilename.fill('0');
			rgbMappedFilename << i;
			rgbMappedFilename << ".tiff";

			cout << "Writing: " << rgbMappedFilename.str() << endl;		
			imwrite(rgbMappedFilename.str().c_str(), rgbMapped);
		}


		// Timestamp
		out << i << "\t" << colorRelTimeArray[i] << endl;

	}

	// Cleaning up
	if(grayBuf) delete [] grayBuf;
	if(depthInColorSpace) delete [] depthInColorSpace;

	ioMutex.lock();
		cout << "WriteColorThread DONE!!" << endl;
		cout << "Color Frames written to " << DUMP_PATH << endl;
		cout << "Color Frames written: " << i << endl;
	ioMutex.unlock();

}

int main()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&kinect);
	if(FAILED(hr)) exit(EXIT_FAILURE);
	
	hr = kinect->Open();
	if(FAILED(hr)) exit(EXIT_FAILURE);

	// Getting coordinate mapper
	hr = kinect->get_CoordinateMapper(&coordMapper);
	if(FAILED(hr)) exit(EXIT_FAILURE);

	thread procDepth(ProcessDepth);
	thread procInfra(ProcessInfra);
	thread procColor(ProcessColor);

	procDepth.join();
	procInfra.join();
	procColor.join();

	cout << "Closing Kinect and cleaning up" << endl;

	hr = kinect->Close();
	if(FAILED(hr)) exit(EXIT_FAILURE);

	SafeRelease(depthReader);
	SafeRelease(infraReader);
	SafeRelease(colorReader);

	cout << "Press s to dump frames to HDD" << endl;
	char c;
	std::cin >> c;

	if(c == 's' || c == 'S') {

		cout << "Dumping to HDD. This could take a while... " << endl;

		thread writeDepth(WriteDepth);
		thread writeInfra(WriteInfra);	
		thread writeColor(WriteColor);

		writeDepth.join();
		writeInfra.join();
		writeColor.join();

		cout << endl;
		cout << "ALL DONE!! Enjoy your K4Wv2 Dump" << endl;
	}

	return EXIT_SUCCESS;
}
