//
//  GMM.cpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/2/19.
//

#include "gmm.hpp"



GMM::GMM(){ }

// ***************************************************
// fits the gaussian models to the data
// ***************************************************
void GMM::fit(Mat *_features, const int _classes)
{
    //init global varibles
    features = _features;
    classes = _classes;
    dimensions = _features->cols;
    no_of_data_samples = features->rows;
    responsibility_mat = Mat::zeros(no_of_data_samples, classes, CV_64FC1);//init r_ic
    
    
    initModels();
    cout << "Start to train " << classes << " models." << endl;
    runEM();
    models_are_trained = true;
    cout << "Finished training!" << endl;
}

// ***************************************************
// predict
// ***************************************************
const vector<int> GMM::predict( Mat *test_data) 
{
    if(!models_are_trained)
    {
        cerr << "No trained models for prediction!" << endl;
        return vector<int>(0);
    }

    vector<int> result;
    for(int r=0; r<test_data->rows; r++)
        result.push_back(findColMaxIndex(test_data->row(r)));
    

    return result;
}

// ***************************************************
// findColMaxIndex
// ***************************************************
int GMM::findColMaxIndex(Mat data_sample)
{
    double max=numeric_limits<int>::min(), value;
    int index;
    for(int c=0; c < classes; c++)
    {
        value = models[c].weight * multiVariateGaussianLog(models[c], data_sample);
        if(value > max)
        {
            index=c;
            max = value;
        }
    }
    return index;
}

// ***************************************************
// saveModels
// ***************************************************
void GMM::saveModels(string path)
{
    ofstream file;
    file.open (path);
     if(!models_are_trained)
    {
        cerr << "No trained models to save" << endl;
        return;
    }

    if(!file.is_open()) return;

    file<<"data_dimensions"<<"="<<dimensions<<endl;
    file<<"classes"<<"="<<classes<<endl;
    for(int i=0; i < classes; i++)
    {
         file<<"model"<<"="<<i<<endl;
         file<<"weight"<<"="<<models[i].weight<<endl;;
         file<<"mean"<<"="<<createStringFromMat(models[i].mean)<<endl;
         file<<"covar"<<"="<<createStringFromMat(models[i].covar)<<endl;
    }
    
    file.close();

}

// ***************************************************
// createStringFromMat
// ***************************************************
string GMM::createStringFromMat(Mat cv_mat)
{
    string mat_string;
    for(int r=0; r < cv_mat.rows; r++)
    {
        for(int c=0; c < cv_mat.cols; c++)
        {
           // mat_string + boost::lexical_cast<std::string>(cv_mat.at<double>(r,c)) +",";
           if(r > 0 || c > 0)
            mat_string.append(",");
           mat_string.append(boost::lexical_cast<std::string>(cv_mat.at<double>(r,c)));
        }
    }

    return mat_string;
}

// ***************************************************
// loadTrainedModels
// ***************************************************
void GMM::loadTrainedModels(string path)
{
    ifstream file(path);
    if (file.is_open())
    {
        models.clear();
        std::string line;
        while(getline(file, line)){
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace),
                                 line.end());
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find("=");
            string name = line.substr(0, delimiterPos);
            string value = line.substr(delimiterPos + 1);
            process_model_load_line(name, value);
        }
        
    }
    else {
        cerr << "Couldn't open config file "<< path <<" for reading."<<endl;
        return;
    }

    models_are_trained = true;
    file.close();
    cout << "Loaded models" << endl;
    
}

// ***************************************************
// process_model_load_line
// ***************************************************
void GMM::process_model_load_line(string name, string value)
{
    if(name == "data_dimensions")
        dimensions = stoi(value);
    else if(name == "classes")
        classes = stoi(value);
    else if(name == "model")
        models.push_back(GaussianModel());
    else if(name=="weight")
        models.back().weight = stod(value);
    else if(name=="mean")
        models.back().mean = createMatFromString(value,dimensions, 1);
    else if(name=="covar")
        models.back().covar = createMatFromString(value, dimensions, dimensions);
    else
    {
        cerr << "Unknown parameter in model load file: "<< name <<endl;
        return;
    }
    

}

// ***************************************************
// createMatFromString
// ***************************************************
Mat GMM::createMatFromString(string in_string, int rows, int cols)
{
    vector<string> string_vec;
    vector<double> float_vec;
    Mat out(rows, cols, CV_64FC1);
    int index=0;

    boost::algorithm::split(string_vec, in_string, boost::is_any_of(","));
    float_vec.resize(string_vec.size());
    std::transform(string_vec.begin(), string_vec.end(), float_vec.begin(), [](const std::string& val){ return stod(val); });
    for(int r=0; r < rows; r++)
        for(int c=0; c<cols; c++)
            out.at<double>(r,c) = float_vec[index++];

   return out;
}

// ***************************************************
// runs the expectation maximization
// ***************************************************
void GMM::runEM()
{
    double diff = std::numeric_limits<double>::infinity();
    double previous_log_likelihood = std::numeric_limits<double>::infinity();
    int it_count = 1;
    while(diff > ITERATION_THR)
    {
        gmmEStep();
        gmmMStep();
        diff = abs(previous_log_likelihood - log_likelihood());
        previous_log_likelihood = log_likelihood();
    }
}


// ***************************************************
// initializes the models
// uses opencv kmeans function to find good starting point
// for the EM algorithm. Kmean start points are found through
// kmeans++
// ***************************************************
void GMM::initModels()
{
    Mat best_labels;
    int max_iterations = 5;
    double required_epsilon = 1.0;
    int kmean_attemps = 50;
    int label;
    
    vector<Mat*> labeled_data(classes);
    for(auto it = labeled_data.begin(); it != labeled_data.end(); ++it)
        *it = new Mat;
    

    Mat dst;
    features->convertTo(dst, CV_32FC1);
    kmeans(dst, classes, best_labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, max_iterations, required_epsilon),
           kmean_attemps, KMEANS_PP_CENTERS );
    
    
    for(int i = 0; i < best_labels.rows; i++)
    {
        label = best_labels.at<int>(i,0);
        labeled_data[label]->push_back(features->row(i));
    }
    
    //init models
    double weigth;
    Mat mean, covariance;
    
    
    for(int i =0; i < classes; i++)
    {
        //weight
        weigth = labeled_data[i]->rows/ static_cast<double>( features->rows );
      
        //mean and covariance
        calcCovarMatrix(*labeled_data[i], covariance, mean, cv::COVAR_ROWS + cv::COVAR_NORMAL + cv::COVAR_SCALE);
        models.push_back(GaussianModel(mean.t(),covariance,weigth));       
    }
    
    //clean memory
    for(auto it = labeled_data.begin(); it != labeled_data.end(); ++it)
        delete *it; 
}

// ***************************************************
// performs E step
// Calculate for each datapoint x_i the probability r_ic
// that datapoint xi belongs to cluster c.
// ***************************************************
void GMM::gmmEStep()
{

    Mat compLogL = datasetLikelihoodLog();
    Mat mixtureLogL = calcMixtureModelsSumLog();
    
    for (int data_i = 0; data_i < no_of_data_samples; data_i++) {
        for (int model_i = 0; model_i < classes; model_i++) {
            responsibility_mat.at<double>(data_i, model_i) = exp(log(models[model_i].weight) +
                                                         compLogL.at<double>(data_i, model_i) -
                                                         mixtureLogL.at<double>(data_i));
            
        }
    }


}


// ***************************************************
// performs M-step
// to calculate the responsibilities
// ***************************************************
void GMM::gmmMStep()
{
    double total_weight_mc;
    double res=0;

    for(uint i=0; i < models.size(); i++)
    {
        total_weight_mc = 0;
        for(int j=0; j < no_of_data_samples; j++)
            total_weight_mc += responsibility_mat.at<double>(j,i);
        
        models[i].weight = total_weight_mc/no_of_data_samples;
        
        
        //mean
        for(int dim = 0; dim < dimensions; dim++)
        {
            for(int data_i=0; data_i < no_of_data_samples; data_i++)
                res +=  responsibility_mat.at<double>(data_i,i)*features->at<double>(data_i,dim);
           
            models[i].mean.at<double>(dim,0) = res/total_weight_mc;
            res = 0;
        }
            

        //covariance matrix
        Mat sigma = Mat::zeros(features->cols, features->cols, CV_64FC1);
        for(int data_i =0; data_i < no_of_data_samples; data_i++)
        {
            Mat transpose_mult_result;
            mulTransposed(features->row(data_i) - models[i].mean.t(), transpose_mult_result, true);
            sigma += transpose_mult_result*responsibility_mat.at<double>(data_i, i);
        }
        models[i].covar = sigma/total_weight_mc;
        
    }
    
}


// ***************************************************
// log-likelihood function of our model
// ***************************************************
double GMM::log_likelihood()
{
    return cv::sum(calcMixtureModelsSumLog()).val[0];
}

// ***************************************************
// datasetLikelihoodLog
// ***************************************************
Mat GMM::datasetLikelihoodLog()
{
    Mat result(no_of_data_samples,classes, CV_64FC1);
    for(int data_i = 0; data_i < no_of_data_samples; data_i++)
        for(int model_i = 0; model_i < classes; model_i++)
            result.at<double>(data_i,model_i) = multiVariateGaussianLog(models[model_i], features->row(data_i) );
    
    return result;

}

// ***************************************************
// calcMixtureLog
// computes evidence for Bayesian formula in E step
// is also used to compute log likelihood
// returns vector including evidence of each data sample
// ***************************************************
Mat GMM::calcMixtureModelsSumLog()
{
    Mat result = Mat::zeros(features->rows, 1, CV_64FC1);
    Mat compLogL = datasetLikelihoodLog();
    double val;
    
    for (int i = 0; i < no_of_data_samples; i++) {
        val = 0;
        for (int j = 0; j < classes; j++) {
            val += exp(compLogL.at<double>(i, j)) * models[j].weight;
        }
        result.at<double>(i,0) = log(val);
    }
    
    double maxLog= -numeric_limits<double>::infinity();;
    for(int i=0; i<result.rows; i++)
        maxLog=max(maxLog,result.at<double>(i,0));
    
    
    for(int i=0; i<result.rows; i++)
        result.at<double>(i,0) = maxLog+log(exp(result.at<double>(i,0)-maxLog));
    
    
    return result;

}

// ***************************************************
// obtain probability density of given datapoint
// compute log[ N(x_i|mu,espilon) ]
// ***************************************************
double GMM::multiVariateGaussianLog(GaussianModel &model, Mat data_sample /* const int dataindex*/) const
{
    double value;
    Mat invertDst;
    invert(model.covar, invertDst);
    Mat transposeDst;
    transpose(data_sample.t() - model.mean, transposeDst);
  

    Mat probability = -0.5 * log(determinant(model.covar))
    - 0.5 * transposeDst * invertDst * (data_sample.t() - model.mean);
    value = probability.at<double>(0, 0);
    
    return value;
   
}

