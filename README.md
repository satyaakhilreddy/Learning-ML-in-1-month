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

# Week 1
## Math Basics
1. [Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
2. [Numpy](https://github.com/jrjohansson/scientific-python-lectures/blob/master/Lecture-2-Numpy.ipynb)
3. [Pandas](https://github.com/ritchieng/pandas-guides)
4. [Matplotlib](https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python)
5. [Math & stats](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch01_Machine_Learning_Basics/NLP%2C%20Math%20%26%20Stats%20Examples.ipynb)  

 
# Week 2 
## Preprocessing
1. [Data Exploration](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/)  
2. [Feature Engineering](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)  
3. [Feature Engineering made easy book](https://github.com/PacktPublishing/Feature-Engineering-Made-Easy)  
4. [Feature selection](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch04_Feature_Engineering_and_Selection/Feature%20Selection.ipynb)
5. [Dimensionality reduction](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/)


# Week 3
## Supervised Learning
[Types of regression](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)  
1. [Linear Regression Overview](https://machinelearningmastery.com/linear-regression-for-machine-learning/) 
    * [Gradient descent explained with an example](https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1)
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
     * [How does regression tree split](https://stats.stackexchange.com/questions/220350/regression-trees-how-are-splits-decided)
6. [Support vector machine](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)

# Week 4
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
    * [Math behind XGBoost](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/) 
    * [Early stopping to avoid overfitting](https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/)
    * [Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)


[Difference between parameter and hyperparameter](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)  
[Train/Test Split and Cross Validation in Python](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)  
[Overfitting vs Underfitting](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

## Unsupervised
[Intro to Clustering and clustering methods](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/)  

## Metrics
1. [Error Metrics](https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/)
2. [Evaluation metrics for classsification](https://towardsdatascience.com/evaluation-metrics-for-classification-409568938a7d)

[Loss functions](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c) 


# Final steps
[Approaching any ML problem](http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/)  

# Ranking problem
## 1. RecSys
Prerequisite: [Singular valued decomposition](https://fenix.tecnico.ulisboa.pt/downloadFile/3779576344458/singular-value-decomposition-fast-track-tutorial.pdf)  
1. [Recommender systems blog](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed)  
2. RecSys, part of Stanford CS246 Mining Massive Datasets : 
   * [1. Overview of Recommender Systems video](https://www.youtube.com/watch?v=hOQg2LQM4ec) 
   * [2. Content based Recommendations ](https://www.youtube.com/watch?v=IlqnNWuqToo)
   * [3. Collaborative Filtering ](https://www.youtube.com/watch?v=3Sl_nFQbLQA)
   * [4. Implementing Collaborative Filtering](https://www.youtube.com/watch?v=Tsmom3S7zeE)
   * [5. Evaluating Recommender systems](https://www.youtube.com/watch?v=Tsmom3S7zeE) 
   * [6. Slides of the above lecture videos part1](http://www.mmds.org/mmds/v2.1/ch09-recsys1.pdf)
   * [7. Sildes of the above lecture videos part2](http://www.mmds.org/mmds/v2.1/ch09-recsys2.pdf)
   * [8. RecSys chapter from the book based on CS246](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)
3. [Netflix RecSys Lecture part 1&2](https://www.youtube.com/watch?v=bLhq63ygoU8)
4. [Netflix RecSys Lecture part 3&4](https://www.youtube.com/watch?time_continue=2&v=mRToFXlNBpQ)
5. [Slides of the above lecture](https://www.slideshare.net/xamat/recommender-systems-machine-learning-summer-school-2014-cmu?ref=http://technocalifornia.blogspot.com/2014/08/introduction-to-recommender-systems-4.html)  
6. [Dimensionality reduction chap from the book based on CS246](http://infolab.stanford.edu/~ullman/mmds/ch11.pdf)


## 2. Search
1. [End to End Search slides](http://www.mices.co/mices2018/slides/Duncan-Blythe_Zalando-Deep_Learning.pdf)
2. [Query understanding in Amazon Search](https://www.slideshare.net/SessionsEvents/tanvi-motwani-lead-data-scientist-guided-search-at-a9com-at-mlconf-atl-2016)
