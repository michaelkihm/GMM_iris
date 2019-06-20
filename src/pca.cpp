//
//  pca.cpp
//  GMM_mnist
//
//  Created by Michael Kihm on 5/2/19.
//  Copyright © 2019 MK. All rights reserved.
//

#include "pca.hpp"

PrincipalComponentAnalysis::PrincipalComponentAnalysis(const Mat *_dataset):dataset(_dataset)
{
    
}


void PrincipalComponentAnalysis::fit(const int _number_of_dimensions)
{
    if(_number_of_dimensions >= dataset->cols)
    {
        cerr << "given dataset has less dimensions then given number of pca's" << endl;
        exit(0);
    }
    
    
    PCA pca(*dataset,Mat(),PCA::DATA_AS_ROW,_number_of_dimensions);
  
    pca.project(*dataset,projection_mat);
    
    //cout << "output size pca: " << projection_mat.size<<endl;
}

Mat PrincipalComponentAnalysis::getProjectedDataSet() const
{
    projection_mat.convertTo(projection_mat,CV_64FC1);
    return projection_mat;
}
