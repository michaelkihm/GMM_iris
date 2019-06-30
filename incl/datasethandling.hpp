//
//  datasethandling.hpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/1/19.
//  Copyright © 2019 MK. All rights reserved.
//
// includes functions which includes helper function for the MNIST dataset

#ifndef datasethandling_hpp
#define datasethandling_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <math.h>
#include "opencv2/opencv.hpp"


enum class FeatureScalar {NORMALIZE, STANDARDIZE};

using namespace cv;
using namespace std;

//global
//cv::Size IMAGE_SIZE = Size(28,28);

//void plotMNISTimage(Mat *dataset, int image_nr);
//Mat datasetRowToImage(Mat *dataset, int row);
vector<Mat> splitFeaturesAndLabels(Mat *dataset, uint feature_col_range[2], uint label_column);
void scaleFeatures(Mat *dataset, uint feature_col_range[2],FeatureScalar method);
void normalizeData(Mat *dataset, uint feature_col_range[2]);
double findMax(Mat *dataset_colum);
double findMin(Mat *dataset_colum);
void standardizeData(Mat *dataset, uint feature_col_range[2]);
double calcMean(Mat feature_vector);
double calcStddev(Mat feature_vector, double mean);
//void labelsToIntegers(Mat *labels);
//void splitTrainTest(Mat *dataset, float test_ratio);


#endif /* datasethandling_hpp */
