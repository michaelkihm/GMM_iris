//
//  GMM.hpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/2/19.
//  Copyright © 2019 MK. All rights reserved.
//
// Guassian Mixture Models Classifier

#ifndef GMM_hpp
#define GMM_hpp

#include <stdio.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include <boost/algorithm/string.hpp>
#include "boost/lexical_cast.hpp"
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>



#include "gmm_data_structs.h"


using namespace std;
using namespace cv;


#define ITERATION_THR 0.01


class GMM
{
public:
    GMM();
    void fit( Mat *_features, const int _classes);
    const vector<int> predict( Mat *test_data);
    void saveModels(string path);
    void loadTrainedModels(string path);
    
private:
    Mat *features;
    int classes;
    int dimensions; //dimension of dataset
    int no_of_data_samples;
    vector<GaussianModel> models;
    Mat responsibility_mat;
    bool models_are_trained;
    
    
    //methods
    string createStringFromMat(Mat cv_mat);
    void process_model_load_line(string name, string value);
    Mat createMatFromString(string in_string,int rows, int cols);
    void runEM();
    void initModels();
    void gmmEStep();
    void gmmMStep();
    double log_likelihood();
    Mat calcMixtureModelsSumLog();
    Mat calcMixtureModels( Mat *data_set_features);
    Mat datasetLikelihoodLog();
    double multiVariateGaussianLog(GaussianModel &model, Mat data_sample) const;
    int findColMaxIndex(Mat data_sample);
    
};


#endif /* GMM_hpp */
