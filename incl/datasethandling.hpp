//
//  datasethandling.hpp
//  GMM_mnist
//
//  Created by Michael Kihm on 5/1/19.
//  Copyright © 2019 MK. All rights reserved.
//
// includes functions which includes helper function for the MNIST dataset

#ifndef datasethandling_hpp
#define datasethandling_hpp

#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//global
//cv::Size IMAGE_SIZE = Size(28,28);

void plotMNISTimage(Mat *dataset, int image_nr);
Mat datasetRowToImage(Mat *dataset, int row);

#endif /* datasethandling_hpp */
