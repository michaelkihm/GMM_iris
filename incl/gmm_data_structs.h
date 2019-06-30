//
//  gmm_data_structs.h
//  GMM_iris
//
//  Created by Michael Kihm on 5/2/19.
//

#ifndef gmm_data_structs_h
#define gmm_data_structs_h

#include "opencv2/opencv.hpp"

struct GaussianModel
{
    cv::Mat mean;
    cv::Mat covar;
    double weight;
    GaussianModel(const cv::Mat _mean, const cv::Mat _covar, double _weight)
    :mean(_mean), covar(_covar), weight(_weight) { };
    GaussianModel() { };
};


#endif /* gmm_data_structs_h */
