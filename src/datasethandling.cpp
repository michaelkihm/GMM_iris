//
//  datasethandling.cpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/1/19.
//  Copyright © 2019 MK. All rights reserved.
//

#include "datasethandling.hpp"


// ***************************************************
// splitFeaturesAndLabels
// ***************************************************
vector<Mat> splitFeaturesAndLabels(Mat *dataset, uint feature_col_range[2], uint label_column)
{
    vector<Mat> out;
    //Mat labels = dataset->col(label_column);
    Mat labels = Mat(dataset->rows, 1, CV_64FC1);
    for(int i=0; i<dataset->rows; i++)
        labels.at<double>(i,0) = dataset->at<double>(i, label_column);

    int col = 0;
   // Mat features = dataset->col(feature_col_range[0]);
    Mat features = Mat(dataset->rows, feature_col_range[1] - feature_col_range[0], CV_64FC1 );
    for(uint c=feature_col_range[0]; c<feature_col_range[1]; c++)
    {
     for(int r=0; r < dataset->rows; r++)
        features.at<double>(r,col) = dataset->at<double>(r,c);
     col +=1;
    }
   
    out.push_back(features);
    out.push_back(labels);
    return out;

}

// ***************************************************
// scaleFeatures
// ***************************************************
void scaleFeatures(Mat *dataset, uint feature_col_range[2],FeatureScalar method)
{
   
    if(!dataset) return; //error
    

    switch(method)
    {
        case FeatureScalar::NORMALIZE:
            normalizeData(dataset, feature_col_range);
            break;
        case FeatureScalar::STANDARDIZE:
            standardizeData(dataset, feature_col_range);
            break;
        default:
            cerr << "Unknown scaling method for scaleFeautures function" << endl;
            break;
    }
}

// ***************************************************
// scaleFeatures
// ***************************************************
void normalizeData(Mat *dataset, uint feature_col_range[2])
{
    double max, min;
    Mat feature_vector;
    if(!dataset) return; //error

    for(uint feature=feature_col_range[0]; feature < feature_col_range[1]; feature++)
    {
        feature_vector = dataset->col(feature);
        max = findMax(&feature_vector);
        min = findMin(&feature_vector);
        for(int data_sample=0; data_sample < dataset->rows; data_sample++)
            dataset->at<double>(feature,data_sample) = (dataset->at<double>(feature,data_sample) - min)/(max-min);
    }

}

// ***************************************************
// findMax
// ***************************************************
double findMax(Mat *dataset_colum)
{
    double max=numeric_limits<double>::min();
  

    for(int i=0; i < dataset_colum->rows; i++)
        if(dataset_colum->at<double>(i,0) > max)
            max = dataset_colum->at<double>(i,0);

    return max;
}

// ***************************************************
// findMin
// ***************************************************
double findMin(Mat *dataset_colum)
{
    double min=numeric_limits<double>::max();

    for(int i=0; i < dataset_colum->rows; i++)
        if(dataset_colum->at<double>(i,0) < min)
            min = dataset_colum->at<double>(i,0);
    
    return min;
}

// ***************************************************
// scaleFeatures
// ***************************************************
void standardizeData(Mat *dataset, uint feature_col_range[2])
{
    double mean, stddev;
    for(uint feature=feature_col_range[0]; feature < feature_col_range[1]; feature++)
    {
        mean = calcMean(dataset->col(feature));
        stddev = calcStddev(dataset->col(feature), mean);
        for(int data_sample=0; data_sample < dataset->rows; data_sample++)
            dataset->at<double>(data_sample,feature) = (dataset->at<double>(data_sample,feature) - mean)/stddev;
    }
                        
}

// ***************************************************
// calcMean
// ***************************************************
double calcMean(Mat feature_vector)
{
    double sum=0;
    for(int i=0; i < feature_vector.rows; i++)
        sum += feature_vector.at<double>(i,0);

    return sum/feature_vector.rows;

}

// ***************************************************
// calcStdev
// ***************************************************
double calcStddev(Mat feature_vector, double mean)
{
    double sum=0, variance;
    for(int i=0; i < feature_vector.rows; i++)
        sum += pow(feature_vector.at<double>(i,0) - mean, 2);

    variance = sum/feature_vector.rows;
    return sqrt(variance);

}

// ***************************************************
// labelsToIntegers
// ***************************************************
// void labelsToIntegers(Mat *labels)
// {
//     vector<string> strings;
//     if(!labels) return; //error

//     for(uint i=0; i<labels->rows; i++)



// }


