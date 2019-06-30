//
//  main.cpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/1/19.
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
  
    //read dataset
    string path = "dataset/Iris2.csv";
    CSVReader csv_reader(path);
    Mat data = csv_reader.getDataset();
    
   
    //process data
    uint feature_cols[2]= {0,4}, label_row = 4;
    scaleFeatures(&data, feature_cols,FeatureScalar::STANDARDIZE);
    vector<Mat> splited_data = splitFeaturesAndLabels(&data, feature_cols, label_row);
    Mat features = splited_data[0];
    Mat labels = splited_data[1]; 
  
    //PCA
    const int no_principal_components = 2;  
    PrincipalComponentAnalysis pca(&features);
    pca.fit(no_principal_components);
    Mat pca_data = pca.getProjectedDataSet();
    
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


    for(int i=0; i < labels.rows; i++)
        if(predictions[i] == label_int.at<short>(i,0))
            sum += 1;

    accuracy = (double)sum/labels.rows;
    cout << "accuracy: " << accuracy << endl;
    
    cout << "FINISHED" << endl;
    return 0;
}
