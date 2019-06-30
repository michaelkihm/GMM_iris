//
//  CSVReader.hpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/1/19.
//  Copyright © 2019 MK. All rights reserved.
//
// modified version of class described at page https://thispointer.com/how-to-read-data-from-a-csv-file-in-c/

#ifndef csvreader_hpp
#define csvreader_hpp

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;

class CSVReader
{
private:
    std::string fileName;
    std::string delimeter;
    
    
public:
    CSVReader(std::string filename, std::string delm = ",") :
    fileName(filename), delimeter(delm)
    { }
    
    // Function to fetch data from a CSV File
    Mat getDataset();
};


#endif /* csvreader_hpp */
