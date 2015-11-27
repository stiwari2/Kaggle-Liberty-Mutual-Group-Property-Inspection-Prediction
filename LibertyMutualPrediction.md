---
author: "Siddhant"
output: html_document
---
```{r}
# Liberty Mutual Insurance is a fortune 100 company. 
# Our task was to predict a transformed count of hazards or pre-existing damages 
# using a dataset of property information. This will enable Liberty Mutual to more accurately identify 
# high risk homes that require additional examination to confirm their insurability. 

##################################################################################################################
####################################### Phase 1: Solving the Regression problem ##################################
##################################################################################################################

#loading the required packages
require(ggplot2)
require(lattice)
require(Matrix)
require(caret)
require(randomForest)
require(gbm)
require(xgboost)
require(ROCR)
df1 = read.csv("train.csv")
summary(df1)
str(df1)

####################################### priliminary data explorarion #############################################


# 50999 observations of 34 variables; 18 numeric columns including ID and targetcolumn Hazard;  
# 17 categorical variables; T1_v16 has the max number of levels (18)


names(df1)
# variable names are anonymized becoz of the business confidentiality constraints of the Liberty group


for (Var in names(df1)) {
  missing = sum(is.na(df1[,Var]))
  if (missing > 0) {
    print(c(Var,missing))
  }
}
#none of the variables have any missing values


#histogram
hist(df1$Hazard, density=100, breaks=40, freq=TRUE, xlab="Hazard", ylab="Frequency", main="Distribution")
axis(side=1, at=seq(0,70,2), labels=seq(0,70,2))    #the distribution approximates -ve exponential curve


#the numeric columns are extracted into a separate dataframe and then their correlation matrix was plotted
numcols = c()
for (i in 1:ncol(df1)) {
    if (class(df1[,i])=="integer") {
       numcols = c(numcols,i)
    }  
}
df_num = df1[,numcols]
df_cat = df1[,-numcols]
str(df_num)
df_num$Id = NULL

cor(df_num)
#some high correlations were noted among independent variables
#T2_V14 and T2_V6 have cor = 0.62; T2_V15 and T2_V4 have cor = 0.5;

levelplot(cor(df_num))


############################################## #train-eval 80:20 split #################################################

set.seed(1)  
N = sample(nrow(df1),round(.8*nrow(df1)),replace=F)

#ID variable is removed from train and eval sets so as to ensure that the models dont get confused
df.train = df1[N,-1]
df.eval = df1[-N,-1]

row.names(df.train) = 1:nrow(df.train)
row.names(df.eval) = 1:nrow(df.eval)


######################################### Evaluation Metric ######################################################

## Evaluation parameter N.Gini index was provided by kaggle
SumModelGini = function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df = df[order(df$submission, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos = sum(df$solution)
  df$cumPosFound = cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz = df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini = df$Lorentz - df$random # will store Lorentz minus random
  print(df)
  return(sum(df$Gini))
}
NormalizedGini = function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}


########################################## Modeling ############################################################

## Linear Regression on whole train set

set.seed(1)
mod1 = lm(Hazard~.,df.train)
summary(mod1)
mod1.rmse = sqrt(mean(mod1$residuals^2)) 
pred1 = predict(mod1,df.eval)
score1 = NormalizedGini(pred1,df.eval$Hazard)
#linear regression achieved Adj R2 =  0.09055, RMSE = 3.824677 and normalized Gini score = 0.3021996

#stepwise subsetting by reducing AIC
step(mod1,trace=T)  

set.seed(1)
mod11 = lm(Hazard~T1_V1+T1_V2+T1_V3+T1_V4+T1_V5+T1_V7+T1_V8+T1_V11+T1_V12+T1_V14+T1_V15+T1_V16+T1_V17+T2_V1+T2_V2+T2_V5+T2_V8+
         T2_V9+T2_V11+T2_V12+T2_V13+T2_V14+T2_V15, df.train)
summary(mod11)  
mod11.rmse = sqrt(mean(mod11$residuals^2))   
pred11 = (predict(mod11,df.eval))
score11 = NormalizedGini(pred11,df.eval$Hazard)  
#LM after stepwise subsetting achieved Adj. R2 = 0.08962, RMSE = 3.824898  and normalized Gini score = 0.3023215

########################################### Modelling ###########################################################

### Random Forest with default mtry
t1 = Sys.time()
set.seed(1)
mod2a = randomForest(Hazard ~ ., df.train, ntree=101)
time_taken = Sys.time() - t1
#Time difference of 2.668209 hours; optimizing mtry is not possible on this dataset coz of the huge temporal cost

mod2a.rmse = sqrt(mean(mod2a$residuals^2))
pred2a_hazard = (predict(mod2a,df.eval))
score2a = NormalizedGini(pred2a_hazard,df.eval$Hazard)  
#Random Forest with default mtry and only 101 trees RMSE = 3.824677 and normalized Gini score = 0.3144161


# GBM was tried on 3 different machines with different values of parameters; crashed every time; hence XGboost
# XGboost with random values of parameters; like GBM, XGboost also requires matrix
train.mx1 = sparse.model.matrix (Hazard ~., df.train)
test.mx1 = sparse.model.matrix (Hazard ~. , df.eval)
dtrain1 = xgb.DMatrix(train.mx1, label=df.train$Hazard)
dtest1 = xgb.DMatrix(test.mx1,  label=df.eval$Hazard)
set.seed(1)
mod3a = xgb.train(params=list(objective='reg:linear', eta=0.01, max_depth=10),data=dtrain1, 
                 nrounds=500, watchlist=list(eval=dtest1, train=dtrain1), maximize=FALSE) 
pred3a = predict(mod3a, newdata=dtest1)
score3a = NormalizedGini(pred3a, df.eval$Hazard)
#the XGboost model achieved RMSE=3.791 and normalized Gini score=0.3450219; best amongst all the models tried out so far


####################### Feature Engineering to reduce the size of the data ####################################

names(df_num)
df_num$Hazard = NULL

#PCA on numeric variables
df_pca = princomp(df_num)  
summary(df_pca)  
#shows that with 9 components out of 16, 98% of variation can be explained

plot(df_pca)
predict(df_pca)
df_num_pca = as.data.frame(predict(df_pca)[,1:9])   
#creating the new dataset with 9 components


str(df_cat)   
# merging categorical levels in variables with more than 5 categories
df_cat$T1_V4 <- as.character(df_cat$T1_V4)
df_cat$T1_V4 <- as.factor(ifelse(df_cat$T1_V4 == "E","Other",
                                   ifelse(df_cat$T1_V4 == "S","Other",
                                     ifelse(df_cat$T1_V4 == "G","Other",
                                            ifelse(df_cat$T1_V4 == "H","Other",df_cat$T1_V4)))))
table(df_cat$T1_V4)


df_cat$T1_V5 <- as.character(df_cat$T1_V5)
df_cat$T1_V5 <- as.factor(ifelse(df_cat$T1_V5 == "I", "Other",
                                 ifelse(df_cat$T1_V5 == "B","Other",
                                     ifelse(df_cat$T1_V5 == "J","Other",
                                            ifelse(df_cat$T1_V5 == "D","Other",
                                                   ifelse(df_cat$T1_V5 == "E","Other",
                                                          ifelse(df_cat$T1_V5 == "L","Other",df_cat$T1_V5)))))))
table(df_cat$T1_V5)                                    


df_cat$T1_V9 <- as.character(df_cat$T1_V9)
df_cat$T1_V9 <- as.factor(ifelse(df_cat$T1_V9 == "C","Other",
                          ifelse(df_cat$T1_V9 == "G","Other",df_cat$T1_V9)))
table(df_cat$T1_V9)


df_cat$T1_V11 <- as.character(df_cat$T1_V11)
df_cat$T1_V11 <- as.factor(ifelse(df_cat$T1_V11 == "F","Other",
                               ifelse(df_cat$T1_V11 == "M","Other",
                                     ifelse(df_cat$T1_V11 == "E","Other",
                                            ifelse(df_cat$T1_V11 == "N","Other",
                                                   ifelse(df_cat$T1_V11 == "D","Other",
                                                          ifelse(df_cat$T1_V11 == "K","Other",
                                                               ifelse(df_cat$T1_V11 == "A","Other",
                                                                    ifelse(df_cat$T1_V11 == "I","Other",df_cat$T1_V11)))))))))
table(df_cat$T1_V11)                                    


df_cat$T1_V15 <- as.character(df_cat$T1_V15)
df_cat$T1_V15 <- as.factor(ifelse(df_cat$T1_V15 == "H","Other",
                                             ifelse(df_cat$T1_V15 == "W","Other",
                                                    ifelse(df_cat$T1_V15 == "S","Other",
                                                           ifelse(df_cat$T1_V15 == "F","Other",df_cat$T1_V15)))))
table(df_cat$T1_V15)                                    


df_cat$T1_V16 <- as.character(df_cat$T1_V16)
df_cat$T1_V16 <- as.factor(ifelse(df_cat$T1_V16 == "A","Other",
                      ifelse(df_cat$T1_V16 == "E","Other",
                         ifelse(df_cat$T1_V16 == "J","Other",
                              ifelse(df_cat$T1_V16 == "N","Other",
                                ifelse(df_cat$T1_V16 == "D","Other",    
                                   ifelse(df_cat$T1_V16 == "M","Other",
                                      ifelse(df_cat$T1_V16 == "C","Other",
                                             ifelse(df_cat$T1_V16 == "L","Other",
                                                    ifelse(df_cat$T1_V16 == "G","Other",
                                                           ifelse(df_cat$T1_V16 == "P","Other",
                                                                ifelse(df_cat$T1_V16 == "H","Other",
                                                                    ifelse(df_cat$T1_V16 == "Q","Other",
                                                                        ifelse(df_cat$T1_V16 == "F","Other",
                                                                           ifelse(df_cat$T1_V16 == "O","Other",df_cat$T1_V16)))))))))))))))
table(df_cat$T1_V16)                                    


df_cat$T2_V5 <- as.character(df_cat$T2_V5)
df_cat$T2_V5 <- as.factor(ifelse(df_cat$T2_V5 == "E","Other",
                                       ifelse(df_cat$T2_V5 == "F","Other",df_cat$T2_V5)))
table(df_cat$T2_V5)                                    


#creating the modified dataset (much smaller and with lesser categories in variables)
df_new = cbind(df_num_pca,df_cat,df1$Hazard)
colnames(df_new)[26] = "Hazard"
write.csv(df_new,"train_reduced.csv",row.names = F)
summary(df_new)
# 10 numeric variables including target and 16 categorical variable
df_new$Hazard = as.numeric(df_new$Hazard)

df.new.train = df_new[N,]
df.new.eval = df_new[-N,]
row.names(df.new.train) = 1:nrow(df.new.train)
row.names(df.new.eval) = 1:nrow(df.new.eval)


################################ Models tuned on this modified data #######################################

# LM on the newly feature engineered data
set.seed(1)
mod41 = lm(Hazard~.,df.new.train)

mod41.rmse = sqrt(mean(mod41$residuals^2)) 
pred41_hazard = round(predict(mod41,df.new.eval))
score41 = NormalizedGini(pred41_hazard,df.new.eval$Hazard)  
# On the whole dataset, normalized min_gini score was 0.3032597
# On the modified dataset, normalized min_gini score achieved was 0.3004685
# This implies that inspite of being much smaller, the new dataset can accurately represent the original

step(mod41, trace = F)
mod41step = lm(formula = Hazard ~ Comp.1 + Comp.2 + Comp.3 + Comp.5 + Comp.6 + 
     Comp.8 + T1_V4 + T1_V5 + T1_V7 + T1_V8 + T1_V9 + T1_V11 + 
     T1_V12 + T1_V15 + T1_V16 + T2_V5 + T2_V11 + T2_V13, data = df.new.train)
summary(mod41step)
mod41step.rmse = sqrt(mean(mod41step$residuals^2)) 
pred41step_hazard = round(predict(mod41step,df.new.eval))
score41step = NormalizedGini(pred41step_hazard,df.new.eval$Hazard) 


# RF on the newly feature engineered data
t1 = Sys.time()
set.seed(1)
mod42a = randomForest(Hazard ~ ., df.new.train, mtry=4, nodesize=10, ntree=501)
time_taken = Sys.time() - t1
#Time difference of 2.172322 hours

pred42a_hazard = round(predict(mod42a,df.new.eval))
score42a = NormalizedGini(pred42a_hazard,df.new.eval$Hazard)  
#On the original dataset, it took 2.668209 hours with only 101 trees; here with 501 trees 1.763627 hours
#Considerable improvement, but not enough to tune mtry manually with all possible values
#So we tried with: 
# mtry=4, score=0.3130904; Time difference of 2.172322 hours ------------ most optimal
# mtry=8, score=0.3039864; Time difference of 2.088691 hours
# mtry=12, score=0.3050804; Time difference of 2.944878 hours
# these runs are removed from the final code for time constraints


# XG-Boost tuning on the newly feature engineered data; tuning portion commented out as time taken is close to 6 hours
# the optimal values obtained from tuning will be used henceforth
train.mx11 = sparse.model.matrix (Hazard ~., df.new.train)
test.mx11 = sparse.model.matrix (Hazard ~. , df.new.eval)
dtrain11 = xgb.DMatrix(train.mx11, label=df.new.train$Hazard)
dtest11 = xgb.DMatrix(test.mx11,  label=df.new.eval$Hazard)

## Create a Grid; 3*3*5=45 combinations
xgb_grid_1 = expand.grid( nrounds = c(1500,2000,5000),
                          eta = c(0.005,0.001,0.0004),
                          max_depth = c(4,5,6,7,8)
)

## Pack the training control parameters
xgb_trcontrol_1 = trainControl( method = "cv",number = 5)

# Train the model for each parameter combination in the grid, using CV to evaluate
t1 = Sys.time()
xgb_train_1 = train(  Hazard ~.  ,trControl = xgb_trcontrol_1,
                      tuneGrid = xgb_grid_1,
                      method = "xgbTree",
                      data=df.new.train,
                      metric='RMSE'
)
time_taken = Sys.time - t1
#Time difference of 5.367219 hours

xgb_train_1$bestTune
xgb_train_1$results
## nrounds=2000,  max_depth=4,  eta=0.005

# Using the optimal value of the parameters on the modified dataset
set.seed(1)
mod43 = xgb.train(params=list(objective='reg:linear', eta=0.005, max_depth=4),data=dtrain11, 
                  nrounds=2000, watchlist=list(eval=dtest11, train=dtrain11), maximize=FALSE)  
pred43_hazard = round(predict(mod43, newdata=dtest11))
score43 = NormalizedGini(pred43_hazard, df.new.eval$Hazard)  
#XGB achieved a normalized mean gini score of 0.3117779 after using the optimal parameters on the modified dataset


########################### Models with optimal parameters were run on original data #############################

#RF with optimal mtry=4 on the original data
t1 = Sys.time()
set.seed(1)
mod2 = randomForest(Hazard ~ ., df.train, mtry=4, nodesize=10, ntree=501, do.trace=T)
time_taken = Sys.time() - t1  
#Time difference of 4.868269 hours

plot(mod2)

#Graphs to visualize importance o the different predictors
varImpPlot(mod2)
imp = importance(mod2)
featureImportance = data.frame(Feature=row.names(imp), Importance=imp[,1])
ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("Importance") +
  ylab("") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

pred2 = predict(mod2,df.eval)
score2 = NormalizedGini(pred2, df.eval$Hazard)  
# By using optimal mtry(=4) and ntree=501 on the original data, 
# RF achieved a normalized mean gini score of 0.3370621 and RMSE=3.803


#XGboost on the original data with optimal parameters
set.seed(1)
mod3 = xgb.train(params=list(objective='reg:linear', eta=0.005, max_depth=4),data=dtrain1, 
                 nrounds=2000, watchlist=list(eval=dtest1, train=dtrain1), maximize=FALSE)  
pred3 = predict(mod3, newdata=dtest1)
score3 = NormalizedGini(pred3, df.eval$Hazard) 
# By using optimal eta(=0.005), optimal max_depth(=4) and optimal nrounds(=2000) on the original data, 
# XGboost achieved a normalized mean gini score of 0.3500576 and RMSE=3.665


#ensembles of RF and XGB with trial and equal weights
e2 = (0.5*pred2 + 0.5*pred3)
score_e2 = NormalizedGini(e2,df.eval$Hazard)  
# normalized mean gini score = 0.3517773; 
# the score achieved by the ensemble was more than the individual scores achieved by both the models


#loop for optimizing the weights in ensemble
score = 0
wt_lm = 0
wt_rf = 0
wt_xgb = 0

for(i in seq(0,1,0.05))  {
  for (j in seq(0,1-i,0.05))  {
    k = 1-i-j
    e = i*pred11 + j*pred2 + k*pred3
    s = NormalizedGini(e, df.eval$Hazard)
    if (s>score)  {
      score = s
      wt_lm = i
      wt_rf = j
      wt_xgb = k
      pred = e
    }
  }
}
# normalized mean gini score = 0.3517773; BEST


output = data.frame(pred11,pred2,pred3,pred,df.eval$Hazard)
write.csv(output, "prediction.csv", row.names = F)
