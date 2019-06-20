//
//  gmm_data_structs.h
//  GMM_mnist
//
//  Created by Michael Kihm on 5/2/19.
//  Copyright © 2019 MK. All rights reserved.
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
};


#endif /* gmm_data_structs_h */
