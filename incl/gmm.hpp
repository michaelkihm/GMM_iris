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
#include <cmath>
#include <algorithm>

#include "gmm_data_structs.h"


using namespace std;
using namespace cv;


#define ITERATION_THR 0.01

//using ulong = unsigned long ;

class GMM
{
public:
    GMM();
    void fit( Mat *_features, Mat *_targets,const int _classes);
    const Mat predict( Mat *test_data);
    void saveModels();
    void loadTrainedModels();
    
private:
    Mat *features;
    Mat *targets;
    int classes;
    int dimensions; //dimension of dataset
    int no_of_data_samples;
    vector<GaussianModel> models;
    Mat responsibility_mat;
    bool models_are_trained;
    
    
    //methods
    void runEM();
    void initModels();
    void gmmEStep();
    void gmmMStep();
    double log_likelihood();
    Mat calcMixtureModelsSumLog();
    Mat calcMixtureModels( Mat *data_set_features);
    Mat datasetLikelihoodLog();
    double multiVariateGaussianLog(GaussianModel &model,const int dataindex) const;
    double multiVariateGaussian(GaussianModel &model,const int dataindex) const;
    
 
    
    
};


#endif /* GMM_hpp */
