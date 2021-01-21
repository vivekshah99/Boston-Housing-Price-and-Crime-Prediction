# Boston-Housing-Price-and-Crime-Prediction

# Introduction

This project is based on the Boston housing dataset. The data was initially collected by Harrison and Rubinfeld for the purpose of examining whether the air quality had any influence over the values of houses in Boston. 
However, in this report, we’re going to be using the Boston Housing dataset to predict the median house prices as well as determine which are the best explanatory variables to determine the median house values in Boston. This report also seeks to identify the best variables to determine per capita crime rate in Boston using other variables. We’ll be using R-programming language in this report to conduct the analyses.

This report is organized in way such that we can demonstrate the entire process right from getting and cleaning the data, to performing exploratory analysis on the dataset, to training Machine Learning models, evaluating those ML models, to performing multiple regression and classification techniques in order to understand the distribution and the importance of several attributes and how they can help in predicting house values as well as the crime rate. In this project we also implement several decision trees algorithms (Random Forest, Gradient Boosting Machine, XGBoost) and run a prediction test to check which decision tree gives us the best split.

# Conclusion

The goal of this report is to determine the neighbourhood attributes that best explain the variation in the house pricing. We used various statistical techniques to eliminate the predictors and to build the most accurate model. Based on our analysis, below are some of the key takeaways from our findings:

1. With this project, we have established that a house value suffers greatly from crime rate, especially suburbs with particularly high crime rate have median house value only half of the median of Boston.

2. This project has also helped us establish that the crime rate per capita is significantly influenced by the rad (accessibility to radial highways), tax and lstat (the wealth level) predictors. These predictors have the strongest positive correlation with crim.

3. The median price for houses where the crime rate was < 1.09025 was found to be $24,290.

4. To predict the median house prices, rm (number of rooms) and lstat are the most important parameters according to our findings. The house prices are significantly higher with more number of rooms and also in areas with low crime. Higher the crime rate, lesser the median value of the houses. Chas i.e. distance to Charles River is also a predictor that somewhat influences the median value of the house prices.

5. The simple linear regression model that was fitted also suggested an important factor, nox (pollution level) plays a key role in determining the pricing of the house. Higher the pollution level in the locality, more the impact its going to have on the house pricing.

6. Using linear regression, we successfully minimized the residual values.

This project provides us with a strong working knowledge on some of the major ML concepts like regression, classification, ensemble methods, tuning methods etc. and also gives great practical hands-on experience in implementing all of the above mentioned concepts.
