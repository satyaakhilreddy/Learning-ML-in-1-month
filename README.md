# Machine learning study flow

## ML Pipeline
1. Data collection (Kafka, spark)
2. Exploratory Data analysis (analyze the input)
3. Feature Engineering (New features/ Derived input)
4. Feature reduction ( Reducing unwanted features)
5. Model building (  F( input ) = Y)
6. Model evaluation ( F(test) = Y* )
7. Metrics to test

## Data Exploration
1. Variable Identification
2. Univariate analysis
    a. **Categorical**:  
       i) We use **frequency table** to understand distribution of each category.  
       ii) It can be measured using two metrics **count & count %**. For visualization: Bar Chart  
    b. **Continuous**:  
       i) **Central tendency** (Mean, Median, Mode) and **spread** of the variable( range, Quartile, IQR, Variance, SD, Skewness, kurtosis)  
       ii) For visualization: Histogram, Box plot  
3. Bi-Variate Analysis ( Relationship between 2 variables)  
    a. **Continuous + Continuous**:  
       i) **Scatter plot**: Shows relationship b/w the variables  
       ii) Correlation: To find the strength of relationship ( +1 perfect positive linear correlation, -1 perfect negative linear     correlation, 0 No correlation)  
    b. **Categorical + Categorical**:   
       1. i) **Two way table**. We can start analyzing relationship by creating a two table of count and count%  
          ii) For visualization: stacked column chart  
       2. **Chi-square test**: Used to derive statistical significance of relationship between variables  
       3. **Cramers V for nominal categorical variable**  
    c. **Continuous + Categorical**:  
       i) Z-test/ t-test  
 
## Preprocessing
1. [Data Exploration](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/)  
2. [Feature Engineering](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)  
3. [Feature Engineering made easy book](https://github.com/PacktPublishing/Feature-Engineering-Made-Easy)  

## Math Basics
1. [Math & stats](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch01_Machine_Learning_Basics/NLP%2C%20Math%20%26%20Stats%20Examples.ipynb)  
2. [Pandas](https://github.com/ritchieng/pandas-guides)
## [Feature selection](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch04_Feature_Engineering_and_Selection/Feature%20Selection.ipynb)  
## [Real world case studies](https://github.com/dipanjanS/practical-machine-learning-with-python/tree/master/notebooks#part-iii-real-world-case-studies)

## [Approaching any ML problem](http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/)  
## [Evaluation metrics for classsification](https://towardsdatascience.com/evaluation-metrics-for-classification-409568938a7d)
## [Support Vector Machines](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)  
## [Overfitting vs Underfitting](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)  
## [Intro to Clustering and clustering methods](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/)  
## [Loss functions](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)  
## [Decision trees](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)  
## []()
