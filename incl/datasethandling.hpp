//
//  datasethandling.hpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/1/19.
//
// includes functions which includes helper function for the Iris dataset

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

vector<Mat> splitFeaturesAndLabels(Mat *dataset, uint feature_col_range[2], uint label_column);
void scaleFeatures(Mat *dataset, uint feature_col_range[2],FeatureScalar method);
void normalizeData(Mat *dataset, uint feature_col_range[2]);
double findMax(Mat *dataset_colum);
double findMin(Mat *dataset_colum);
void standardizeData(Mat *dataset, uint feature_col_range[2]);
double calcMean(Mat feature_vector);
double calcStddev(Mat feature_vector, double mean);


#endif /* datasethandling_hpp */
