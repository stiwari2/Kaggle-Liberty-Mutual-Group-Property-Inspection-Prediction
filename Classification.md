```
###################################################################################################################
############################### Phase-2: Converting it into a classification Problem ##############################
###################################################################################################################

# After solving the problem given in kaggle, I looked at the problem from a different angle.
# Liberty Mutual Insurance want to identify high risk homes that require additional examination 
# to confirm their insurability. So basically they have a set threshold. If the predicted hazard value is more
# than this threshold, they will examine the property again. So instead of forecasting the exact Hazard value,
# we can simply predict if the Hazard score is more than a given threshold. This makes it a binomial 
# classification problem. For simplicity we assumed the threshld as the 3rd quartile of the hazard score (=5)
# from the train data.


## creating the classifier , removing ID and Hazard score
df1$Hazard1=ifelse(df1$Hazard>5,1,0)
df1$Hazard1 = as.factor(df1$Hazard1)
df1$Hazard=NULL
df1$Id=NULL


## train-eval 80:20 split 
set.seed(1)  
N = sample(nrow(df1),round(.8*nrow(df1)),replace=F)
df.train = df1[N,]
df.eval = df1[-N,]
row.names(df.train) = 1:nrow(df.train)
row.names(df.eval) = 1:nrow(df.eval)



##Logistic Regression
mod.1 = glm(Hazard1~.,data=df.train,family=binomial(link='logit'))
summary(mod.1)
pred.glm=predict(mod.1,type='response',df.eval)
head(pred.glm)
length(pred.glm)
pred.glm1=ifelse(pred.glm>0.5,1,0)

#Misclassification
conf.matrix=table(df.eval$Hazard1,pred.glm1)
#Looking at the business case, False Negetive is the most important error metric here and should be as less as possible
#In this model, the FN count is quite high making it a poor model

MisclassificationRGLM = 1-sum(diag(conf.matrix))/sum(conf.matrix)
## 0.2325 is the Misclassification rate achieved from Logistic Regression



## Random forest - Using default mtry for classifiicaiton
set.seed(1)
cprop.rf <- randomForest(Hazard1 ~ ., data=df.train,mtry=sqrt(32), ntree=500,do.trace = TRUE)
plot(cprop.rf)

# prediction on test data
cprob.rf <- predict(cprop.rf,df.eval,type='class')
cprob.rf1 <- predict(cprop.rf,df.eval,type='prob')[,2]

# misclassification error in test data
conf.matrix2=table(df.eval$Hazard1,cprob.rf)
#again FN is quite high

MisclassificationRrf = 1-sum(diag(conf.matrix2))/sum(conf.matrix2)
## 0.2339 is the Misclassification rate achieved from RF



## XGboost 
train =df.train
test = df.eval
train$Hazard1 = as.numeric(train$Hazard1) - 1
test$Hazard1 = as.numeric(test$Hazard1) - 1
train.mx=sparse.model.matrix(Hazard1~., train)
test.mx=sparse.model.matrix(Hazard1~., test)
dtrain=xgb.DMatrix(train.mx, label=train$Hazard1)
dtest=xgb.DMatrix(test.mx, label=test$Hazard1)
train.gdbt=xgb.train(params=list(objective='binary:logistic', eta=0.05, max_depth=2), data=dtrain, 
                      nrounds=2700, watchlist=list(eval=dtest, train=dtrain))

cprop.predxg=predict(train.gdbt,newdata=dtest)
cprop.xgb=ifelse(cprop.predxg>0.5,1,0)
conf.matrix3=table(df.eval$Hazard1,cprop.xgb)
MisclassificationRxg = 1-sum(diag(conf.matrix3)/sum(conf.matrix3))
## 0.228 is the Misclassification rate achieved from XGboost



#ROC curve and AUC of the 3 models
prediction.lm <- prediction(pred.glm, df.eval$Hazard1)
err.lm <- performance(prediction.lm, measure = "err")
prediction.rf <- prediction(cprob.rf1, df.eval$Hazard1)
err.rf <- performance(prediction.rf, measure = "err")
prediction.xgb <- prediction(cprop.predxg, df.eval$Hazard1)
err.xgb <- performance(prediction.xgb, measure = "err")

plot(err.lm, ylim=c(0.1, 0.9))
plot(err.rf, col=2, add=TRUE)
plot(err.xgb, col=3, add=TRUE)

ROC.lm <- performance(prediction.lm, measure = "tpr", x.measure = "fpr")
ROC.rf <- performance(prediction.rf, measure = "tpr", x.measure = "fpr")
ROC.xgb <- performance(prediction.xgb, measure = "tpr", x.measure = "fpr")

plot(ROC.lm)
abline(a=0, b=1, lty=2) # diagonal line
plot(ROC.rf, add=T, col="blue")
plot(ROC.xgb, add=T, col="red")

as.numeric(performance(prediction.lm, "auc")@y.values)
as.numeric(performance(prediction.rf, "auc")@y.values)
as.numeric(performance(prediction.xgb, "auc")@y.values)
#AUC_logistic =  0.6709261; AUC_rf = 0.6792559; AUC_xgb = 0.6950479
```

