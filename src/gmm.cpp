//
//  GMM.cpp
//  GMM_iris
//
//  Created by Michael Kihm on 5/2/19.
//  Copyright © 2019 MK. All rights reserved.
//

#include "gmm.hpp"



GMM::GMM(){ }

// ***************************************************
// fits the gaussian models to the data
// ***************************************************
//Mat GMM::predict(Mat &test_data)

void GMM::fit(Mat *_features, Mat *_targets,const int _classes)
{
    //init global varibles
    features = _features;
    targets = _targets;
    classes = _classes;
    dimensions = _features->cols;
    no_of_data_samples = features->rows;
    responsibility_mat = Mat::zeros(no_of_data_samples, classes, CV_64FC1);//init r_ic
    
    
    initModels();
    cout << "Start to train " << classes << " models." << endl;
    runEM();
    cout << "Finished training!" << endl;
    
}

// ***************************************************
// predict
// ***************************************************
const Mat GMM::predict( Mat *test_data) 
{
    return Mat(0,0,CV_8U);
}

// ***************************************************
// saveModels
// ***************************************************
void GMM::saveModels()
{
    
}

// ***************************************************
// loadTrainedModels
// ***************************************************
void GMM::loadTrainedModels()
{
    
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
        cout << "iteration no: " << it_count++ << endl;
        gmmEStep();
        gmmMStep();
        diff = abs(previous_log_likelihood - log_likelihood());
        previous_log_likelihood = log_likelihood();
        cout << "iteration likelihood: " << previous_log_likelihood << endl;
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
    int max_iterations = 10;
    double required_epsilon = 1.0;
    int kmean_attemps = 5;
    int label;
    
    vector<Mat*> labeled_data(classes);
    for(auto it = labeled_data.begin(); it != labeled_data.end(); ++it)
        *it = new Mat;
    
    //cout << "init models using kmeans" << endl;
    Mat dst;
    features->convertTo(dst, CV_32FC1);
    kmeans(dst, classes, best_labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, max_iterations, required_epsilon),
           kmean_attemps, KMEANS_PP_CENTERS );
    
    
    //iterate over clusters and assign each data point to the found labels
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

        //cout << "\n"<<covariance.inv() << endl;

        models.push_back(GaussianModel(mean.t(),covariance,weigth));
        
    }
    
    //cout << models[0].covar << endl;yy
    //clean memory
    for(auto it = labeled_data.begin(); it != labeled_data.end(); ++it)
        delete *it;//*it = new Mat;
    //labeled_data.clear();
    
}

// ***************************************************
// performs E step
// Calculate for each datapoint x_i the probability r_ic
// that datapoint xi belongs to cluster c.
// ***************************************************
void GMM::gmmEStep()
{
//    double sum = 0;
//    for(int data_index=0; data_index < probability_mat.rows; data_index++)
//    {
//        for(int model_index = 0; model_index < probability_mat.cols; model_index++)
//        {
//            
//            for(int i=0; i < probability_mat.cols; i++)
//                sum += models[i].weight * multiVariateGaussian(models[i], data_index);
//            
//            probability_mat.at<float>(data_index,model_index) =
//            models[model_index].weight* multiVariateGaussian(models[model_index], data_index) / sum;
//            sum = 0;
//                
//        }
//    }
    //cout << "E step"<<endl;
    
    //multiVariateGaussian(GaussianModel model, Mat datapoint)
    //Mat result = Mat::zeros(model.size(), features.cols, CV_32FC1);
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
    //cout << "M step"<<endl;
    double total_weight_mc;
    double res=0;
    //Mat result;
    
    
    //double weight_pi_c;
    //double mean_mu_c;
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
            //res;//result.at<float>(0,0);
            

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
//    double result = 0;
//    double inner_sum = 0;
//    for(int data_i = 0; data_i < no_of_data_samples; data_i++){
//        for(int model_i=0; model_i < classes; model_i++)
//            inner_sum += models[model_i].weight * multiVariateGaussian(models[model_i], data_i);
//        result += log(inner_sum);
//        inner_sum = 0;
//    }
//
//    return result;
    return cv::sum(calcMixtureModelsSumLog()).val[0];
}

// ***************************************************
//
// ***************************************************
Mat GMM::datasetLikelihoodLog()
{
    Mat result(no_of_data_samples,classes, CV_64FC1);
    for(int data_i = 0; data_i < no_of_data_samples; data_i++)
        for(int model_i = 0; model_i < classes; model_i++)
            result.at<double>(data_i,model_i) = multiVariateGaussianLog(models[model_i], data_i);
    
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
double GMM::multiVariateGaussianLog(GaussianModel &model, const int dataindex) const
{
      double value;
      Mat datapoint = features->row(dataindex).t();
//    double constant =  pow(2*M_PI,dimensions/2);
//    double sigma_det = pow(determinant(model.covar),0.5);
//    Mat exp_term(1,1,CV_32FC1);
//
//    Mat test = Mat(datapoint-model.mean).t();
//    Mat t2 = datapoint-model.mean;
//    Mat t7 = test * t2;
//    cout << "stuff"<<endl;
//    cv::exp(-0.5* Mat(datapoint-model.mean).t()  *model.covar.inv() * datapoint-model.mean, exp_term);
//
//    return pow(constant*sigma_det,-1)*exp_term.at<float>(0,0);
    
    
    Mat invertDst;
    invert(model.covar, invertDst);
    Mat transposeDst;
    transpose(datapoint - model.mean, transposeDst);
  

    Mat probability = -0.5 * log(determinant(model.covar))
    - 0.5 * transposeDst * invertDst * (datapoint - model.mean);
    value = probability.at<double>(0, 0);
    
    return value;
   
}


// ***************************************************
//  calcMixtureModels
//  computes max_value = 0;max_value = 0;max_value = 0;max_value = 0;max_value = 0;
// ***************************************************
//Mat GMM::calcMixtureModels(Mat *features)
//{
//     Mat compLogL = datasetLikelihoodLog();
//}
