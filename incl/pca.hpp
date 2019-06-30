//
//  pca.hpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/2/19.
//
// principal component analysis
#ifndef pca_hpp
#define pca_hpp

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

class PrincipalComponentAnalysis
{
public:
    PrincipalComponentAnalysis(const Mat *_dataset);
    ~PrincipalComponentAnalysis() { }
    void fit(const int _number_of_dimensions);
    Mat getProjectedDataSet() const; 
private:
    const Mat *dataset;
    Mat projection_mat;
};


#endif /* pca_hpp */
