//
//  CSVReader.cpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/1/19.
//  Copyright © 2019 MK. All rights reserved.
//

#include "csvreader.hpp"

#define DEBUG_NR 6000

Mat CSVReader::getDataset()
{
   
    int counter = 0;
    vector<vector<double>> dataList;
    std::ifstream file(fileName);
    
    if(!file.good())
    {
        cerr << "CANNOT OPEN FILE" << endl;
        exit(0);
    }

    
    std::string line = "";
    vector<string> string_vec;
    vector<double> float_vec;
    // Iterate through each line and split the content using delimeter
    while (getline(file, line) )//&& counter < DEBUG_NR)
    {
        string_vec.clear();
        float_vec.clear();
        boost::algorithm::split(string_vec, line, boost::is_any_of(delimeter));
        //convert to float
        float_vec.resize(string_vec.size());
        std::transform(string_vec.begin(), string_vec.end(), float_vec.begin(), [](const std::string& val){ return stod(val); });

        dataList.push_back(float_vec);
        counter += 1;
    }
    // Close the File
    file.close();
    
    Mat temp((int)dataList.size(), (int)dataList[0].size(), CV_64FC1);//improve with pushback
    for(int r = 0; r < temp.rows; r++)
        for(int c =0; c < temp.cols; c++)
            temp.at<double>(r,c) = dataList[r][c];
    

    return temp;
}
