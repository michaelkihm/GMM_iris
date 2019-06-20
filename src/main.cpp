//
//  main.cpp
//  GMM_mnist
//
//  Created by Michael Kihm on 5/1/19.
//  Copyright © 2019 MK. All rights reserved.
//

#include <iostream>
#include "datasethandling.hpp"
#include "csvreader.hpp"
#include "gmm.hpp"
#include "pca.hpp"
#include <string>

using namespace std;


int main(int argc, const char * argv[]) {
    // insert code here...
    string path = "/Users/michaelkihm/Desktop/train_images_mnist.csv";
    string test_feature_path = "/Users/michaelkihm/Desktop/test_images_mnist.csv";
    string test_label_path = "/Users/michaelkihm/Desktop/train_labels_mnist.csv";
    //string path_label = "/Users/michaelkihm/Desktop/train_labels_mnist.csv";
    //string path = "/Users/michaelkihm/Desktop/test.csv";
    CSVReader csv_reader(path);
    Mat data = csv_reader.getDataset();
    
    //CSVReader csv_label_reader(path_label);
    //Mat labels_data = csv_label_reader.getDataset();
    const int no_principal_components = 40;
    
    PrincipalComponentAnalysis pca(&data);
    pca.fit(no_principal_components);
    GMM gmm;

    Mat t = pca.getProjectedDataSet();
    
    
    
    

    
    
    
   // gmm.fit(&t, NULL, 10);
    
    
    
    CSVReader test_feature_reader(test_feature_path);
    Mat test_features = test_feature_reader.getDataset();
    
    CSVReader test_label_reader(test_label_path);
    Mat test_label = test_label_reader.getDataset();
    
    PrincipalComponentAnalysis pca_test_features(&test_features);
    pca_test_features.fit(no_principal_components);
    
    Mat pca_data = pca_test_features.getProjectedDataSet();
   // Mat predictions = gmm.predict(&pca_data);
    
    
    Mat  dst, best_labels;
    int max_iterations = 10;
    double required_epsilon = 1.0;
    int kmean_attemps = 5;;
    t.convertTo(dst, CV_32FC1);
    kmeans(dst, 10, best_labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, max_iterations, required_epsilon),
           kmean_attemps, KMEANS_PP_CENTERS );
    
    //check accurancy
    int sum = 0;
    for(int i = 0; i < test_label.rows; i++)
    {
        //int label = (int)test_label.at<double>(i,0);
        //int predict = best_labels.at<int>(i,0);
        //cout << label << " " << predict << endl;
        if(best_labels.at<int>(i,0) == (int)test_label.at<double>(i,0))
            sum += 1;
        
    }
    cout << "accurancy: " << sum/(double)test_label.rows << endl;
    
//    float d[] = {1, 3, 5, 5, 4, 1, 3, 8, 6 };
//    Mat A(3,3, CV_32F, d);
//    cout << A << endl;
//    Mat cov, mean;
//    
//    Mat c = pca.getProjectedDataSet();
//    Mat fuck ;
//    for(int i = 0; i < 150; i++)
//    {
//        int label = i*2;
//        fuck.push_back(c.row(label));
//    }
//    
//    calcCovarMatrix(fuck, cov, mean, cv::COVAR_ROWS + cv::COVAR_NORMAL + cv::COVAR_SCALE);
//    cout << cov << endl;
//    cout << mean << endl;
//    cout << cov.inv() << endl;
    
    cout << "FINISHED" << endl;
    return 0;
}
