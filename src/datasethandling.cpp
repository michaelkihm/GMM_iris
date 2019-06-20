//
//  datasethandling.cpp
//  GMM_mnist
//
//  Created by Michael Kihm on 5/1/19.
//  Copyright © 2019 MK. All rights reserved.
//

#include "datasethandling.hpp"


void plotMNISTimage(Mat *dataset, int image_nr)
{

    if(image_nr > dataset->rows)
    {
        cerr << "Image nr " << image_nr << " is not in dataset"<< endl;
        return;
    }
    string window_name = "Image " + to_string(image_nr);
    imshow(window_name, datasetRowToImage(dataset,image_nr));
    waitKey();
    
}


Mat datasetRowToImage(Mat *dataset, int row)
{
    Size image_size(28,28);
    Mat image = Mat(image_size,CV_64FC1,dataset->row(row).data);
    image *= 255;
    transpose(image, image);
    return image;
}
