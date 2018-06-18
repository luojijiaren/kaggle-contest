setwd('C:/Users/fzhan/Documents/GitHub/Caravan Insurance Challenge')
data=read.csv('caravan-insurance-challenge.csv')
training=data[which(data$ORIGIN=='train'),][,2:87]
#0=data[which(data$CARAVAN==0),]

#1=data[which(data$CARAVAN==1),]

library(DMwR)
training$CARAVAN=as.factor(training$CARAVAN)
table(training$CARAVAN)
training=SMOTE(CARAVAN~.,training,perc.over=600,perc.under = 117)
table(training$CARAVAN)
#training$CARAVAN=as.numeric(training$CARAVAN)
#s=sample.int(n=nrow(t0),size=nrow(t1),replace=F)
#t2=t0[s,]
#train=rbind(t1,t2)
test=data[which(data$ORIGIN=='test'),][,2:87]


set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 90% of data as sample from total 'n' rows of the data  

sample <- sample.int(n = nrow(training), size = floor(.9*nrow(training)), replace = F)
train <- training[sample, ]
eval  <- training[-sample, ]

attach(training)
library(randomForest)
set.seed(1)
CARAVAN=as.factor(CARAVAN)
tr=randomForest(CARAVAN~.,ntree=100,data=training,subset=sample,mtry=29,importance=TRUE)
yhat.tr=predict(tr,newdata = eval)

pretable=table(yhat.tr,factor(eval$CARAVAN))
accuracy=sum(diag(pretable))/nrow(eval)
accuracy 

#get roc
library(pROC)
auc=roc(eval$CARAVAN,as.numeric(as.character(yhat.tr)))
auc   #0.89

#logistic
glm.fit=glm(CARAVAN~.,data=training,subset=sample,family=binomial)
glm.probs=predict(glm.fit,newdata=eval,type='response')
glm.pred=rep(0,nrow(eval))
glm.pred[glm.probs>.5]=1
auc=roc(eval$CARAVAN,as.numeric(as.character(glm.pred)))
auc   #0.7866

#svm
library(e1071)
svmfit=svm(CARAVAN~.,data=training,subset=sample,kernal='polynomial',cost=12)
yhat.svm=predict(svmfit,newdata = eval)
auc=roc(eval$CARAVAN,as.numeric(as.character(yhat.svm)))
auc    #0.9142

#boosting
library(gbm)
set.seed(4)
boost=gbm(CARAVAN~.,data=training[sample,],distribution='gaussian',n.trees=5000,interaction.depth=4)
yhat.boost=predict(boost,newdata = training[-sample,],n.trees=5000)
auc=roc(eval$CARAVAN,as.numeric(as.character(yhat.boost)))
auc    #0.9443

#gxboost

#ann




