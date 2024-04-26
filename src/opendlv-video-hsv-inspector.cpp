/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <chrono>

std::string getCurrentTime();

int32_t main(int32_t argc, char **argv)
{
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ((0 == commandlineArguments.count("name")) ||
        (0 == commandlineArguments.count("width")) ||
        (0 == commandlineArguments.count("height")))
    {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image and transform it to HSV color space for inspection." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --name=<name of shared memory area> --width=<W> --height=<H>" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "Example: " << argv[0] << " --name=img.argb --width=640 --height=480" << std::endl;
    }
    else
    {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid())
        {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            // Create an OpenCV image header using the data in the shared memory.
            IplImage *iplimage{nullptr};
            CvSize size;
            size.width = WIDTH;
            size.height = HEIGHT;

            iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 4 /* four channels: ARGB */);
            sharedMemory->lock();
            {
                iplimage->imageData = sharedMemory->data();
                iplimage->imageDataOrigin = iplimage->imageData;
            }
            sharedMemory->unlock();

            cv::namedWindow("Inspector", CV_WINDOW_AUTOSIZE);
            int minH_B{105};
            int maxH_B{151};
            int minH_Y{13};
            int maxH_Y{32};
            cvCreateTrackbar("BLUE Hue (min)", "Inspector", &minH_B, 179);
            cvCreateTrackbar("BLUE Hue (max)", "Inspector", &maxH_B, 179);

            cvCreateTrackbar("YELLOW Hue (min)", "Inspector", &minH_Y, 179);
            cvCreateTrackbar("YELLOW Hue (max)", "Inspector", &maxH_Y, 179);

            int minS{84};
            int maxS{255};
            cvCreateTrackbar("Sat (min)", "Inspector", &minS, 255);
            cvCreateTrackbar("Sat (max)", "Inspector", &maxS, 255);

            int minV{39};
            int maxV{255};
            cvCreateTrackbar("Val (min)", "Inspector", &minV, 255);
            cvCreateTrackbar("Val (max)", "Inspector", &maxV, 255);


            // Endless loop; end the program by pressing Ctrl-C.
            while (cv::waitKey(10))
            {
                cv::Mat img;


                // Don't wait for a notification of a new frame so that the sender can pause while we are still inspection
                // sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    // Copy image into cvMat structure.
                    // Be aware of that any code between lock/unlock is blocking
                    // the camera to provide the next frame. Thus, any
                    // computationally heavy algorithms should be placed outside
                    // lock/unlock.
                    img = cv::cvarrToMat(iplimage);
                }
                sharedMemory->unlock();

                double alpha, beta; // declare contrast control and brightness control

                // Adjust contrast and brightness dynamically
                double aveBrightness = cv::mean(img)[0];
                if (aveBrightness < 100)
                {
                    alpha = 1.5; // Increase contrast for darker images
                    beta = 50;   // Increase brightness
                }
                else if (aveBrightness > 180)
                {
                    alpha = 0.8; // Decrease contrast for brighter images
                    beta = -30;  // Decrease brightness
                }
                else
                {
                    alpha = 1.0;
                    beta = 0;
                }
                img.convertTo(img, -1, alpha, beta);

                int videoWidth = img.cols;
                int videoHeight = img.rows;

                int centerLineX = videoWidth / 2;
                int centerLineY = videoHeight / 2;

                // Center Line X: 320 Video width: 640 Video height: 480
                //std::cout << "Center Line X: " << centerLineX << ", Video width: " << videoWidth  << ", Video height: " << videoHeight << std::endl;

                cv::Mat imgHSV;
                cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

                cv::Mat imgColorSpace;

                // Declare an image to store values for blue, and an image to store values for yellow
                cv::Mat imgBlue, imgYellow;

                cv::inRange(imgHSV, cv::Scalar(minH_B, minS, minV), cv::Scalar(maxH_B, maxS, maxV), imgBlue);
                cv::inRange(imgHSV, cv::Scalar(minH_Y, minS, minV), cv::Scalar(maxH_Y, maxS, maxV), imgYellow);

                // Combine the blue and yellow images into one Mat object.
                cv::bitwise_or(imgBlue, imgYellow, imgColorSpace);

                // Create an output image initialized to black (all zeros)
                cv::Mat outputImage = cv::Mat::zeros(img.size(), img.type());

                // Set detected blue areas to blue color (BGR format for blue)
                outputImage.setTo(cv::Scalar(255, 0, 0), imgBlue);

                // Set detected yellow areas to yellow color (BGR format for yellow)
                outputImage.setTo(cv::Scalar(0, 255, 255), imgYellow);

                // Apply morphological operations (erosion and dilation) to reduce noise
                cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
                cv::morphologyEx(outputImage, outputImage, cv::MORPH_OPEN, morphKernel);

                cv::Mat canny_output;
                cv::Canny ( outputImage, canny_output, 100, 200);

                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(canny_output, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
                std::vector<cv::Rect> boundRect( contours.size() );
                std::vector<cv::Point2f>centers( contours.size() );

                for( size_t i = 0; i < contours.size(); i++ )
                {
                    cv::approxPolyDP( contours[i], contours_poly[i], 3, true );
                    boundRect[i] = cv::boundingRect( contours_poly[i] );
                }

                cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

                // Create a blue list and yellow list to store the boundingRects
                std::vector<cv::Rect> blueRects;
                std::vector<cv::Rect> yellowRects;

                // for( size_t i = 0; i< contours.size(); i++ )
                // {
                //     cv::Scalar color = cv::Scalar( 0, 255, 0 );
                //     cv::drawContours( drawing, contours_poly, (int)i, color );
                //     cv::rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
                //     std::cout << "Rectangle " << i << " size: " << boundRect[i].tl() << "x" << boundRect[i].br() << std::endl; 
                // }


                // // Find contours in the filtered image
                // std::vector<std::vector<cv::Point>> contours;
                // std::vector<cv::Vec4i> hierarchy;
                // cv::findContours(imgColorSpace, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


                // Draw bounding rectangles around the contours
                for (size_t i = 0; i < contours.size(); i++)
                {
                    std::string currentTime = getCurrentTime();

                    // Specify the position and size of the rectangle
                    cv::Rect upHalf(0, 0, 640, 240);
                    cv::Rect carBody(180, 380, 300, 100);

                    cv::Scalar color = cv::Scalar( 0, 255, 0 );
                    cv::drawContours( drawing, contours_poly, (int)i, color );
                    cv::rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 1);

                    cv::Rect boundingRect = cv::boundingRect(contours[i]);

                    // Ignore the rectangles at top half and the car's body
                    if(boundingRect.height > 10 && boundingRect.y > centerLineY &&
                        !upHalf.contains(boundingRect.tl()) &&
                        !carBody.contains(boundingRect.tl())){

                        // The color is determined by how the pixels contained in boundingRect are distributed in imgBlue and imgYellow
                        cv::Mat roiBlue = imgBlue(boundingRect);
                        cv::Mat roiYellow = imgYellow(boundingRect);
                        
                        // Count the total number of pixels in the blue section and the yellow section
                        int bluePixels = cv::countNonZero(roiBlue);
                        int yellowPixels = cv::countNonZero(roiYellow);
                        
                        // Judge the color based on the number of pixels
                        if (bluePixels > yellowPixels) {
                            // Push boundingRect to the list
                            blueRects.push_back(boundingRect);
                            std::cout << "Blue cones rectangle: x=" << boundingRect.x << ", y=" << boundingRect.y << ", height=" << boundingRect.height << ", time=" << currentTime << std::endl;
                        } else {
                            // Push boundingRect to the list
                            yellowRects.push_back(boundingRect);
                            std::cout << "Yellow cones rectangle: x=" << boundingRect.x << ", y=" << boundingRect.y << ", height=" << boundingRect.height << ", time=" << currentTime << std::endl;
                        }
                    }
                    
                    

                    // Print rectangle position information
                    std::cout << "Rectangle " << i + 1 << ": x=" << boundingRect.x << ", y=" << boundingRect.y << ", width=" << boundingRect.width << ", height=" << boundingRect.height << ", time=" << getCurrentTime() << std::endl;
                    cv::rectangle(img, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0, 255, 0), 1);

                    // Draw the fixed size rectangle on the image
                    cv::rectangle(img, upHalf, cv::Scalar(0, 0, 255), 3);
                    cv::rectangle(img, carBody, cv::Scalar(0, 0, 255), 3);
                }

                // Find the two boundingrects with the largest height values in the blueRects
                std::partial_sort(blueRects.begin(), blueRects.end(), blueRects.end(), [](const cv::Rect& a, const cv::Rect& b) {
                    return a.height > b.height;
                });

                // Find the two boundingrects with the highest height values in yellowRects
                std::partial_sort(yellowRects.begin(), yellowRects.end(), yellowRects.end(), [](const cv::Rect& a, const cv::Rect& b) {
                    return a.height > b.height;
                });

                // Calculate the distance ratio of the blue rectangle
                double blueDistanceRatio = 0.0;
                if (blueRects.size() >= 2) {
                    int blueHeightSum = blueRects[0].height + blueRects[1].height;
                    int blueCenterSum = std::abs(centerLineX - blueRects[0].x) + std::abs(centerLineX - blueRects[1].x);
                    blueDistanceRatio = static_cast<double>(blueCenterSum) / blueHeightSum / 2.0;
                }

                // Calculate the distance ratio of the yellow rectangle
                double yellowDistanceRatio = 0.0;
                if (yellowRects.size() >= 2) {
                    int yellowHeightSum = yellowRects[0].height + yellowRects[1].height;
                    int yellowCenterSum = std::abs(centerLineX - yellowRects[0].x) + std::abs(centerLineX - yellowRects[1].x);
                    yellowDistanceRatio = static_cast<double>(yellowCenterSum) / yellowHeightSum / 2.0;
                }

                // Print distance ratio
                std::cout << "Blue distance ratio: " << blueDistanceRatio << std::endl;
                std::cout << "Yellow distance ratio: " << yellowDistanceRatio << std::endl;

                cv::imshow("Color-Space Image", drawing);  // Display the combined mask
                cv::imshow(sharedMemory->name().c_str(), img);   // Display the original image
                cv::imshow("Color Filtered Image", outputImage); // Display the output image with highlighted colors

            }

            if (nullptr != iplimage)
            {
                cvReleaseImageHeader(&iplimage);
            }
        }
        retCode = 0;
    }
    return retCode;
}

std::string getCurrentTime() {
    // Gets the current timestamp, expressed as a point in time on the system clock
    auto now = std::chrono::system_clock::now();

    // Convert a point in time to time t
    auto now_c = std::chrono::system_clock::to_time_t(now);

    // Gets the microsecond portion of the current time
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;

    // Convert the time portion to local time
    std::tm tm_now{};
    localtime_r(&now_c, &tm_now);

    // Creates a stream of strings to format the time
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << '.' << std::setw(6) << std::setfill('0') << now_us.count();

    // Return a string time
    return oss.str();
}