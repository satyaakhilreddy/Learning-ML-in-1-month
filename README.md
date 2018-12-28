# Machine learning study flow
[Machine Learning course By Standford University](https://www.coursera.org/learn/machine-learning/home/welcome)  
[Machine Learning Coursera notes](http://www.holehouse.org/mlclass/index.html) 

## ML Pipeline
1. Data collection (Kafka, spark)
2. Exploratory Data analysis (analyze the input)
3. Feature Engineering (New features/ Derived input)
4. Feature reduction ( Reducing unwanted features)
5. Model building (  F( input ) = Y)
6. Model evaluation ( F(test) = Y* )
7. Metrics to test

 
## Preprocessing
1. [Data Exploration](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/)  
2. [Feature Engineering](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)  
3. [Feature Engineering made easy book](https://github.com/PacktPublishing/Feature-Engineering-Made-Easy)  
4. [Feature selection](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch04_Feature_Engineering_and_Selection/Feature%20Selection.ipynb)
5. [Dimensionality reduction](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/)

## Math Basics
1. [Math & stats](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch01_Machine_Learning_Basics/NLP%2C%20Math%20%26%20Stats%20Examples.ipynb)  
2. [Numpy](https://github.com/jrjohansson/scientific-python-lectures/blob/master/Lecture-2-Numpy.ipynb)
3. [Pandas](https://github.com/ritchieng/pandas-guides)
4. [Matplotlib](https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python)
5. [Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)


## Supervised Learning
[Types of regression](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)  
1. [Linear Regression Overview](https://machinelearningmastery.com/linear-regression-for-machine-learning/)  
    * [Evaluation metrics for regression](https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/)
    * [Assumptions of Linear Regression](https://www.statisticssolutions.com/assumptions-of-linear-regression/)
    * [Simple Linear Regression in Python](https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9) 
    
2. [Logistic Regression](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)  
    * [Building logistic regression in Python](https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)
3. [Loss functions for regression](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)
    * [Loss functions implementation in Python](https://nbviewer.jupyter.org/github/groverpr/Machine-Learning/blob/master/notebooks/05_Loss_Functions.ipynb)
4. [Regularization](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)    
     * [Lasso and ridge regression](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/)
5. [Tree based Modeling](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)  
6. [Support vector machine](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)

#### Ensembling Techniques
1. Bagging:  
We build many independent predictors/models/learners and combine them using some model averaging techniques. (e.g. weighted average, majority vote or normal average)
    * [Random forest explained](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd)
       * [Solving a problem using Random Forest](https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/)
2. Boosting:  
An ensemble technique in which the predictors are not made independently, but sequentially.  
    * [Gradient boosting](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
         * Widely used GBM algorithms: 1. XGBoost  2. LightGBM  3. CATBoost
    * [Gradient Boosting from scratch in python](https://www.kaggle.com/grroverpr/gradient-boosting-simplified/)
    * [Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)



[Train/Test Split and Cross Validation in Python](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)
[Overfitting vs Underfitting](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

## Unsupervised
[Intro to Clustering and clustering methods](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/)  

## Metrics
1. [Error Metrics](https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/)
2. [Evaluation metrics for classsification](https://towardsdatascience.com/evaluation-metrics-for-classification-409568938a7d)

[Loss functions](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c) 

[Recommender systems](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed)

[Approaching any ML problem](http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/)  

#
## Data Exploration
1. Variable Identification

2. Univariate analysis

    a. **Categorical**:  
       * We use **frequency table** to understand distribution of each category.  
       * It can be measured using two metrics **count & count %**. For visualization: Bar Chart  
       
    b. **Continuous**:  
       * **Central tendency** (Mean, Median, Mode) and **spread** of the variable( range, Quartile, IQR, Variance, SD, Skewness, kurtosis)  
       * For visualization: Histogram, Box plot  
       
3. Bi-Variate Analysis ( Relationship between 2 variables)

    a. **Continuous + Continuous**:  
       * **Scatter plot**: Shows relationship b/w the variables  
       * **Correlation:** To find the strength of relationship ( +1 perfect positive linear correlation, -1 perfect negative linear     correlation, 0 No correlation)  
       
    b. **Categorical + Categorical**:   
       * **Two way table**. We can start analyzing relationship by creating a two table of count and count%. For visualization: stacked column chart  
       * **Chi-square test**: Used to derive statistical significance of relationship between variables  
       * **Cramers V for nominal categorical variable** 
       
    c. **Continuous + Categorical**:  
       * Z-test/ t-test  
