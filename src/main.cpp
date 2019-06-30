//
//  main.cpp
//  GMM_iris
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
#include <fstream>

using namespace std;


int main(int argc, const char * argv[]) {
  
    string path = "dataset/Iris2.csv";
    CSVReader csv_reader(path);
    Mat data = csv_reader.getDataset();
    
   
    //process data
    uint feature_cols[2]= {0,4}, label_row = 4;
    scaleFeatures(&data, feature_cols,FeatureScalar::STANDARDIZE);
//    cout << "Standard DATA: \n"<<data << endl;
    vector<Mat> splited_data = splitFeaturesAndLabels(&data, feature_cols, label_row);
    Mat features = splited_data[0];
    Mat labels = splited_data[1]; 
    //CSVReader csv_label_reader(path_label);
    //Mat labels_data = csv_label_reader.getDataset();

    //cout << "LABELS"<<labels << endl;
    //cout << "FEATURES"<< features << endl;
    //PCA
    const int no_principal_components = 2;  
    PrincipalComponentAnalysis pca(&features);
    pca.fit(no_principal_components);
    Mat pca_data = pca.getProjectedDataSet();

    // ofstream myfile;
    // myfile.open ("pca.csv");
    // for(int r=0; r < pca_data.rows; r++)
    // {
    //     myfile << pca_data.at<double>(r,0)<<","<<pca_data.at<double>(r,1)<<endl;
    // }
    //  myfile.close();
    
    //GMM
    GMM gmm;
    const int species_classes = 3;
    gmm.fit(&pca_data,species_classes);
    //gmm.saveModels("/home/michael/Desktop/models.txt");
    //gmm.loadTrainedModels("/home/michael/Desktop/models.txt");
    
    double accuracy;
    int sum =0;
    Mat label_int;
    vector<int> predictions = gmm.predict(&pca_data);
    labels.convertTo(label_int,CV_16UC1);

    // cout <<"predictions:" << endl;
    // for(uint i=0; i < predictions.size(); i++)
    //     cout << predictions[i] << endl;
     //cout << "label int: "<<endl << label_int << endl;

    // for(int i =0; i < label_int.rows; i++)
    // {
    //     if(label_int.at<short>(i,0) == 0)
    //         label_int.at<short>(i,0) = 1;

    //     if(label_int.at<short>(i,0) == 1)
    //         label_int.at<short>(i,0) = 0;
    // }

    for(int i=0; i < labels.rows; i++)
        if(predictions[i] == label_int.at<short>(i,0))
            sum += 1;

    accuracy = (double)sum/labels.rows;
    cout << "accuracy: " << accuracy << endl;
    
    

    
    
    
   // gmm.fit(&t, NULL, 10);
    
    
    
    // CSVReader test_feature_reader(test_feature_path);
    // Mat test_features = test_feature_reader.getDataset();
    
    // CSVReader test_label_reader(test_label_path);
    // Mat test_label = test_label_reader.getDataset();
    
    // PrincipalComponentAnalysis pca_test_features(&test_features);
    // pca_test_features.fit(no_principal_components);
    
    // Mat pca_data = pca_test_features.getProjectedDataSet();
   // Mat predictions = gmm.predict(&pca_data);
    
    
    // Mat  dst, best_labels;
    // int max_iterations = 10;
    // double required_epsilon = 1.0;
    // int kmean_attemps = 5;;
    // t.convertTo(dst, CV_32FC1);
    // kmeans(dst, 10, best_labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, max_iterations, required_epsilon),
    //        kmean_attemps, KMEANS_PP_CENTERS );
    
    // //check accurancy
    // int sum = 0;
    // for(int i = 0; i < test_label.rows; i++)
    // {
    //     //int label = (int)test_label.at<double>(i,0);
    //     //int predict = best_labels.at<int>(i,0);
    //     //cout << label << " " << predict << endl;
    //     if(best_labels.at<int>(i,0) == (int)test_label.at<double>(i,0))
    //         sum += 1;
        
    // }
    // cout << "accurancy: " << sum/(double)test_label.rows << endl;
    
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
