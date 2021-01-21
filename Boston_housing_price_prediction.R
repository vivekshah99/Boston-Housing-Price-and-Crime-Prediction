#### CLEAN WORKSPACE ####
rm(list = ls ())

#### LOAD PACKAGES ####

library(MASS)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(boot)
library(dplyr)
library(tidyr)
library(tidyverse)
library(caTools)
library(corrplot)
library(reshape2)
library(gam)
library(randomForest)
library(gbm)
library(tree)
library(yhat)
library(Amelia)
library(xgboost)

########## LOAD DATA ########## 

Boston_housing <- read.csv("HousingData.csv", stringsAsFactors = FALSE)

########## CHECK FOR MISSING VALUES ########## 

sum(is.na(Boston_housing))

########## CHECK FOR DUPLICATE VALUES ########## 

sum(duplicated(Boston_housing))

########## SPLITTING TEST AND TRAINING DATASET ########## 

set.seed(123)

split <- sample.split(Boston_housing, SplitRatio = 0.75)
train <- subset(Boston_housing, split== TRUE)
test <- subset(Boston_housing, split== FALSE)

summary(Boston_housing)

########## DATA CLEANING ########## 

missmap(Boston_housing,col=c('orange','black'),y.at= 1, y.labels= '',legend= TRUE)

##########  DATA EXPLORATION ########## 

names(Boston_housing)
head(Boston_housing)
str(Boston_housing)
summary(Boston_housing)

##########  CREATING NEW COLUMN FOR MEDV ########## 

Boston_housing['MEDV']= Boston_housing

##########  BOX PLOT TO IDENTIFY OUTLIERS AND COMMENT ########## 

par(mfrow= c(1,4))
boxplot(Boston_housing$CRIM, 
        main= 'crim', 
        col = "Pink")

boxplot(Boston_housing$ZN, 
        main= 'zn', 
        col = "Green")

boxplot(Boston_housing$RM, 
        main= 'rm', 
        col = "Orange")

boxplot(Boston_housing$B, 
        main='B', 
        col="Yellow")

boxplot(Boston_housing$MEDV, 
        main='medv', 
        col="Yellow")

##########  CORRELATION PLOT ##########  

corrl_mat <- cor(Boston_housing)
corrplot(corrl_mat, type = "upper")

Boston_housing[1:5,1:5]

dim(Boston_housing)

class(Boston_housing)

corr_mat= cor(Boston_housing,method = "s")
corr_mat[1:5,1:5]
corrplot(cor(Boston_housing))

cor(Boston_housing, Boston_housing$MEDV)

########## LINEAR REGRESSION MODEL ##########  

lm.fit1 <- lm(MEDV~., data = train)
summary(lm.fit1)

lm.fit2 <- lm(MEDV~.-AGE-INDUS+I (LSTAT^2), data = train)
summary(lm.fit2)

lm.fit3 <- lm(MEDV~.-INDUS-AGE-ZN+RM*LSTAT-B+RM*RAD+LSTAT*RAD, data = train)
summary(lm.fit3)

plot(lm.fit1, 
     col= "Pink")

plot(lm.fit2, 
     col= "Sky Blue")

plot(lm.fit3, 
     col= "Green")

##########  SCATTER PLOT ##########  

ggplot(data = melt(Boston_housing, id="MEDV"), aes(x=value, y=MEDV)) +
  facet_wrap(~variable, scales="free") +
  geom_point(color = "black") + 
  ggtitle("Scatterplot of Median value based on all variables of Boston housing dataset") +
  xlab("") + 
  ylab("Per capita median value")

##########  PLOTTING HISTOGRAM FOR HOUSE PRICING ##########  

residuals <- data.frame('Residuals' = lm.fit3$residuals)
res_hist <- ggplot(residuals, 
                   aes(x=Residuals)) + geom_histogram(color='black', fill='skyblue') + ggtitle('Histogram of Residuals')
res_hist

hist(Boston_housing$MEDV, 
     xlab = "Median Value", 
     main = "House Pricing", 
     col = "sky blue")

##########  LOGISTIC REGRESSION ##########  

Boston_housing$RESULT <- ifelse(Boston_housing$MEDV > 30,1,0)
names(Boston_housing)
set.seed(123)
split <- sample.split(Boston_housing$RESULT, SplitRatio = 0.7)
split

train <- subset(Boston_housing, split== TRUE)
test <- subset(Boston_housing, split== FALSE)
train$MEDV <- NULL
test$MEDV <- NULL

logis <- glm(RESULT ~., data = train, family = binomial)
summary(logis)
train$PRED <- predict(logis, data= train, type = "response")
test$PRED <- predict(logis, newdata = test, type = "response")

##########  CONFUSION MATRIX ##########  

table(actual= train$RESULT, 
      predict= train$PRED > 0.2)


##########  REGRESSION TREES ##########  

set.seed(123)
train = sample(1:nrow(Boston_housing)/2)  ### DIVIDING 50:50 i.e. TRAIN AND TEST
tree.Boston_housing= tree (MEDV ~., Boston_housing, subset = train)

summary(tree.Boston_housing)

plot(tree.Boston_housing)

text(tree.Boston_housing, pretty = 0)

head(Boston_housing)

######## CROSS VALIDATION WITH CV.TREE FUNCTION ########

cv.Boston_housing = cv.tree(tree.Boston_housing)
plot(cv.Boston_housing$size, cv.Boston_housing$dev, type= 'b')
ggplot(mapping = aes(x=cv.Boston_housing$size, y=cv.Boston_housing$dev)) + geom_line()


prune.Boston_housing = prune.tree(tree.Boston_housing, best = 5) #to avoid overfitting
plot(prune.Boston_housing)
text(prune.Boston_housing, pretty = 0)

##########  TREE PERFORMANCE TEST ##########  

yhat = predict(tree.Boston_housing, newdata= Boston_housing[-train,])
Boston_housing.test= Boston_housing[-train, "MEDV"]
print(mean((yhat-Boston_housing.test)^2))
plot(yhat,Boston_housing.test)
abline(0,1)

########## RANDOM FORESTS ##########

dim(Boston_housing)

attach(Boston_housing)

set.seed(123)
train<-sample(1:nrow(Boston_housing),300)

rf.Boston_housing = randomForest(MEDV~.,data = Boston_housing, mtry=6, subset = train, importance= TRUE)
rf.Boston_housing
plot(rf.Boston_housing)
yhat.rf= predict(rf.Boston_housing, newdata= Boston_housing[-train, ])
round(mean((yhat.rf-Boston_housing.test)^2),2)
importance(rf.Boston_housing)
varImpPlot(rf.Boston_housing)

########## GBM BOOSTING ##########  

set.seed(42)
boost.Boston_housing = gbm(MEDV~.,data= Boston_housing[train,],distribution= "gaussian", n.trees= 5000, interaction.depth= 4)
summary(boost.Boston_housing)
par(mfrow=c(1,2))

plot(boost.Boston_housing, 
     i="RM")

plot(boost.Boston_housing, 
     i="LSTAT")

yhat.boost = predict(boost.Boston_housing, newdata= Boston_housing[-train,], n.trees= 5000)
round(mean((yhat.boost-Boston_housing.test)^2),2)
Boston.boost <- gbm(MEDV~.,data = Boston_housing[train,],distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.1, verbose = F, cv.folds = 10)

########## FINDING THE BEST TREE FOR PREDICTION ##########  

bestTreeForPrediction <- gbm.perf(Boston.boost)
yhat.boost= predict(Boston.boost, newdata= Boston_housing[-train,],n.trees= bestTreeForPrediction)
round(mean((yhat.boost-Boston_housing.test)^2),2)
outputdataset = data.frame("CRIM"= Boston_housing$CRIM, "MEDV"= Boston_housing$MEDV)

head(outputdataset)

write.csv(outputdataset, "prediction_Boston_housing.csv", row.names = FALSE)

write.csv(Boston_housing,"Boston_hosuing_test.csv",row.names = FALSE)

############### XGBOOST ############### 

train.Boston_housing <- Boston_housing[train,]
test.Boston_housing <- Boston_housing[-train,]

datatrain<- xgb.DMatrix(data = as.matrix(train.Boston_housing[!names(train.Boston_housing)%in% c("MEDV")]), label= train.Boston_housing$MEDV)
Boston_housing.xgb = xgboost(data = datatrain, max_depth= 3, 
                             eta= 0.2, 
                             nthread= 3, 
                             nrounds = 40, 
                             lambda= 0,
                             objective= "reg:linear")
datatest<- as.matrix(test.Boston_housing[!names(train.Boston_housing) %in% c("MEDV")])
yhat.xgb <- predict(Boston_housing.xgb,datatest)
round(mean((yhat.xgb-Boston_housing.test)^2),2)

set.seed(123)
week6 <- list("max_depth"= 3, "eta"=0.2, "objective"= "reg:linear", "lambda"= 0)
cv.ab <- 500
cv.cd <- 3
Boston_housing.xgb <- xgb.cv(week6=week6, 
                             data = datatrain, 
                             nfold = cv.ab, 
                             nrounds = cv.cd, 
                             early_stopping_rounds = 200, 
                             verbose = 0)

datatrain <- xgb.DMatrix(data = as.matrix(train.Boston_housing[!names(train.Boston_housing) %in% c("MEDV")]),label= train.Boston_housing$MEDV)
Boston_housing.xgb = xgboost(week6= week6, data = datatrain, nthread= 3, nrounds = Boston_housing.xgb$best_iteration, verbose = 0)
datatest <- as.matrix(test.Boston_housing[!names(train.Boston_housing) %in% c("MEDV")])
yhat.xgb <- predict(Boston_housing.xgb, datatest)
round(mean((yhat.xgb-Boston_housing)^2),2)

imp <- xgb.importance(colnames(Boston_housing[!names(Boston_housing) %in% c("MEDV")]), model = Boston_housing.xgb)
imp
xgb.plot.importance(imp, rel_to_first = TRUE, 
                    xlab= "Relative Importance")



