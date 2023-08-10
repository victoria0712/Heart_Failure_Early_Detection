setwd('/Users/Victoria/Documents/Projects_Github/Heart Failure Early Detection')
library(caTools)
library(rpart)
library(randomForest)
library(dplyr)
library(rpart.plot)

# data preparation --------------------------------------------------------
df = read.csv("heart_failure_clinical_records_dataset.csv")
summary(df$Heart_failure)
# checking missing values 
sum(is.na(df))

normalize = function (target) {
  (target - min(target))/(max(target) - min(target))
}
df$age = normalize(df$age)
df$creatinine_phosphokinase = normalize(df$creatinine_phosphokinase)
df$ejection_fraction = normalize(df$ejection_fraction)
df$platelets = normalize(df$platelets)
df$serum_creatinine = normalize(df$serum_creatinine)
df$serum_sodium = normalize(df$serum_sodium)
df$Heart_failure = factor(df$Heart_failure)
df$anaemia = factor(df$anaemia)
df$diabetes = factor(df$diabetes)
df$high_blood_pressure = factor(df$high_blood_pressure)
df$sex = factor(df$sex)
df$smoking = factor(df$smoking)
df$time = NULL

# verify no missing 
sum(is.na(df))

# create result table -----------------------------------------------------
table = data.frame('Features' = 1:9, 'Model' = 1:9, 'Accuracy' = 1:9, 
                   'error.rate(false positive)' = 1:9,
                   'error.rate(false negative)' = 1:9)
table[] = lapply(table, as.character)
table[1:3,1] = "all"
table[4:6,1] = "6"
table[7:9,1] = "4"
table[c(1,4,7),2] = "logistic regression"
table[c(2,5,8),2] = "CART"
table[c(3,6,9),2] = "random forest"

# train-test split and oversampling of trainset ---------------------------
set.seed(65)

train = sample.split(Y = df$Heart_failure, SplitRatio = 0.8)
trainset = subset(df, train == T)
testset = subset(df, train == F)
table(trainset$Heart_failure)
table(testset$Heart_failure)
disease = trainset[trainset$Heart_failure == 1,]
n = sum(trainset$Heart_failure == 0)

oversample = sample_n(disease, n, replace = TRUE)
trainset = rbind(trainset[trainset$Heart_failure == 0, ], oversample)
table(trainset$Heart_failure)

# all features: logistic regression ---------------------------------------
logis = glm(Heart_failure ~ ., family = binomial, data = trainset)
summary(logis)
pred.logis = ifelse(predict(logis, newdata = testset, type="response")>0.5,1,0)
cm.l = table(testset$Heart_failure, pred.logis,dnn=c("actual","predict"))   
cm.l
round(prop.table(cm.l, margin = 1),4)
table[1,3] = (cm.l[1,1]+cm.l[2,2])/sum(cm.l)
table[1,4] = cm.l[1,2]/(cm.l[1,1]+cm.l[1,2])
table[1,5] = cm.l[2,1]/(cm.l[2,1]+cm.l[2,2])
paste("accuracy of logistic regression is", (cm.l[1,1]+cm.l[2,2])/sum(cm.l))
paste("error rate (false positive) is", cm.l[1,2]/(cm.l[1,1]+cm.l[1,2]))
paste("error rate (false negative) is", cm.l[2,1]/(cm.l[2,1]+cm.l[2,2]))

# all features: CART ------------------------------------------------------
cart_max = rpart(Heart_failure ~ . , data = trainset, method = 'class',cp=0)
#summary(cart_max)
rpart.plot(cart_max, nn=T)
#print(cart_max)
plotcp(cart_max)
#printcp(cart_max, digits = 4)
pred.cart_max = ifelse(predict(cart_max, newdata = testset)[,2]>0.5,1,0)
cm.c.max = table(testset$Heart_failure,pred.cart_max,dnn=c("actual","predict"))   
cm.c.max
round(prop.table(cm.c.max, margin = 1),4)
cart_max$variable.importance
cart_max.scaledVarImpt <- round(100*cart_max$variable.importance/sum(cart_max$variable.importance))
cart_max.scaledVarImpt

## automate search for optimal tree 
CVerror.cap = cart_max$cptable[which.min(cart_max$cptable[,"xerror"]),"xerror"]+
  cart_max$cptable[which.min(cart_max$cptable[,"xerror"]),"xstd"]
CVerror.cap
i=1;j=4
while (cart_max$cptable[i,j] > CVerror.cap) {
  print(i)
  (i=i+1)}
cp.opt = ifelse(i>1, sqrt(cart_max$cptable[i,1]*cart_max$cptable[i-1,1]),1)
cp.opt
cart_opt = prune(cart_max, cp = cp.opt)
rpart.plot(cart_opt, nn=T)
pred.cart_opt = ifelse(predict(cart_opt, newdata = testset)[,2]>0.5,1,0)
cm.c_opt = table(testset$Heart_failure,pred.cart_opt,dnn=c("actual","predict"))   
cm.c_opt
round(prop.table(cm.c_opt, margin = 1),4)

table[2,3] = (cm.c_opt[1,1]+cm.c_opt[2,2])/sum(cm.c_opt)
table[2,4] = cm.c_opt[1,2]/(cm.c_opt[1,1]+cm.c_opt[1,2])
table[2,5] = cm.c_opt[2,1]/(cm.c_opt[2,1]+cm.c_opt[2,2])
paste("accuracy of CART is", (cm.c_opt[1,1]+cm.c_opt[2,2])/sum(cm.c_opt))
paste("error rate (false positive) is", cm.c_opt[1,2]/(cm.c_opt[1,1]+cm.c_opt[1,2]))
paste("error rate (false negative) is", cm.c_opt[2,1]/(cm.c_opt[2,1]+cm.c_opt[2,2]))

# all features: random forest ---------------------------------------------
rf = randomForest(Heart_failure ~ ., data = trainset, importance = T, keep.inbag = T)
rf
plot(rf)
pred.rf = predict(rf, newdata = testset)
cm.r = table(testset$Heart_failure,pred.rf,dnn=c("actual","predict"))
cm.r
round(prop.table(cm.r, margin = 1),4)
table[3,3] = (cm.r[1,1]+cm.r[2,2])/sum(cm.r)
table[3,4] = cm.r[1,2]/(cm.r[1,1]+cm.r[1,2])
table[3,5] = cm.r[2,1]/(cm.r[2,1]+cm.r[2,2])
paste("accuracy of random forest is",(cm.r[1,1]+cm.r[2,2])/sum(cm.r))
paste("error rate (false positive) is", cm.r[1,2]/(cm.r[1,1]+cm.r[1,2]))
paste("error rate (false negative) is", cm.r[2,1]/(cm.r[2,1]+cm.r[2,2]))
importance(rf)
sort(rf$importance[,4], decreasing = TRUE)
rf.scaledVarImpt <- round(100*rf$importance[,4]/sum(rf$importance[,4]))
rf.scaledVarImpt

# 6 most important features: logistic regression --------------------------
trainset1 = trainset[, c("ejection_fraction","serum_creatinine",
                         "creatinine_phosphokinase","age","serum_sodium","platelets",
                         "Heart_failure")]
testset1 = testset[, c("ejection_fraction","serum_creatinine",
                       "creatinine_phosphokinase","age","serum_sodium","platelets",
                       "Heart_failure")] 


# six features: logistic regression ---------------------------------------
logis1 = glm(Heart_failure ~ ., family = binomial, data = trainset1)
summary(logis1)
pred.logis1 = ifelse(predict(logis1, newdata = testset1, type="response")>0.5,1,0)
cm.l1 = table(testset1$Heart_failure, pred.logis1,dnn=c("actual","predict"))   
cm.l1
round(prop.table(cm.l1, margin = 1),4)
table[4,3] = (cm.l1[1,1]+cm.l1[2,2])/sum(cm.l1)
table[4,4] = cm.l1[1,2]/(cm.l1[1,1]+cm.l1[1,2])
table[4,5] = cm.l1[2,1]/(cm.l1[2,1]+cm.l1[2,2])
paste("accuracy of logis1tic regression is", (cm.l1[1,1]+cm.l1[2,2])/sum(cm.l1))
paste("error rate (false positive) is", cm.l1[1,2]/(cm.l1[1,1]+cm.l1[1,2]))
paste("error rate (false negative) is", cm.l1[2,1]/(cm.l1[2,1]+cm.l1[2,2]))

# six features: CART ------------------------------------------------------
cart_max1 = rpart(Heart_failure ~ . , data = trainset1, method = 'class',cp=0)
pred.cart_max1 = ifelse(predict(cart_max1, newdata = testset1)[,2]>0.5,1,0)
cm.c.max1 = table(testset1$Heart_failure,pred.cart_max1,dnn=c("actual","predict"))   
cm.c.max1
round(prop.table(cm.c.max1, margin = 1),4)

## automate search for optimal tree 
CVerror.cap1 = cart_max1$cptable[which.min(cart_max1$cptable[,"xerror"]),"xerror"]+
  cart_max1$cptable[which.min(cart_max1$cptable[,"xerror"]),"xstd"]
CVerror.cap1
i=1;j=4
while (cart_max1$cptable[i,j] > CVerror.cap1) {
  print(i)
  (i=i+1)}
cp.opt1 = ifelse(i>1, sqrt(cart_max1$cptable[i,1]*cart_max1$cptable[i-1,1]),1)
cp.opt1
cart_opt1 = prune(cart_max1, cp = cp.opt1)
pred.cart_opt1 = ifelse(predict(cart_opt1, newdata = testset1)[,2]>0.5,1,0)
cm.c_opt1 = table(testset1$Heart_failure,pred.cart_opt1,dnn=c("actual","predict"))   
cm.c_opt1
round(prop.table(cm.c_opt1, margin = 1),3)

table[5,3] = (cm.c_opt1[1,1]+cm.c_opt1[2,2])/sum(cm.c_opt1)
table[5,4] = cm.c_opt1[1,2]/(cm.c_opt1[1,1]+cm.c_opt1[1,2])
table[5,5] = cm.c_opt1[2,1]/(cm.c_opt1[2,1]+cm.c_opt1[2,2])
paste("accuracy of CART is", (cm.c_opt1[1,1]+cm.c_opt1[2,2])/sum(cm.c_opt1))
paste("error rate (false positive) is", cm.c_opt1[1,2]/(cm.c_opt1[1,1]+cm.c_opt1[1,2]))
paste("error rate (false negative) is", cm.c_opt1[2,1]/(cm.c_opt1[2,1]+cm.c_opt1[2,2]))


# 6 most important features: random forest --------------------------------
rf1 = randomForest(Heart_failure ~ ., data = trainset1, importance = T, keep.inbag = T)
rf1
plot(rf1)
pred.rf1 = predict(rf1, newdata = testset1)
cm.r1 = table(testset1$Heart_failure,pred.rf1,dnn=c("actual","predict"))
cm.r1
round(prop.table(cm.r1, margin = 1),4)
table[6,3] = (cm.r1[1,1]+cm.r1[2,2])/sum(cm.r1)
table[6,4] = cm.r1[1,2]/(cm.r1[1,1]+cm.r1[1,2])
table[6,5] = cm.r1[2,1]/(cm.r1[2,1]+cm.r1[2,2])
paste("accuracy of random forest is",(cm.r1[1,1]+cm.r1[2,2])/sum(cm.r1))
paste("error rate (false positive) is", cm.r1[1,2]/(cm.r1[1,1]+cm.r1[1,2]))
paste("error rate (false negative) is", cm.r1[2,1]/(cm.r1[2,1]+cm.r1[2,2]))


# 4 most important features: logistic regression --------------------------
trainset2 = trainset[, c("ejection_fraction","serum_creatinine",
                         "creatinine_phosphokinase","age",
                         "Heart_failure")]
testset2 = testset[, c("ejection_fraction","serum_creatinine",
                       "creatinine_phosphokinase","age",
                       "Heart_failure")]


# four features: logistic regression ---------------------------------------
logis2 = glm(Heart_failure ~ ., family = binomial, data = trainset2)
summary(logis2)
pred.logis2 = ifelse(predict(logis2, newdata = testset2, type="response")>0.5,1,0)
cm.l2 = table(testset2$Heart_failure, pred.logis2,dnn=c("actual","predict"))   
cm.l2
round(prop.table(cm.l2, margin = 1),4)
table[7,3] = (cm.l2[1,1]+cm.l2[2,2])/sum(cm.l2)
table[7,4] = cm.l2[1,2]/(cm.l2[1,1]+cm.l2[1,2])
table[7,5] = cm.l2[2,1]/(cm.l2[2,1]+cm.l2[2,2])
paste("accuracy of logis2tic regression is", (cm.l2[1,1]+cm.l2[2,2])/sum(cm.l2))
paste("error rate (false positive) is", cm.l2[1,2]/(cm.l2[1,1]+cm.l2[1,2]))
paste("error rate (false negative) is", cm.l2[2,1]/(cm.l2[2,1]+cm.l2[2,2]))

# four features: CART ------------------------------------------------------
cart_max2 = rpart(Heart_failure ~ . , data = trainset2, method = 'class',cp=0)
pred.cart_max2 = ifelse(predict(cart_max2, newdata = testset2)[,2]>0.5,1,0)
cm.c.max2 = table(testset2$Heart_failure,pred.cart_max2,dnn=c("actual","predict"))   
cm.c.max2
round(prop.table(cm.c.max2, margin = 1),4)

## automate search for optimal tree 
CVerror.cap2 = cart_max2$cptable[which.min(cart_max2$cptable[,"xerror"]),"xerror"]+
  cart_max2$cptable[which.min(cart_max2$cptable[,"xerror"]),"xstd"]
CVerror.cap2
i=1;j=4
while (cart_max2$cptable[i,j] > CVerror.cap2) {
  print(i)
  (i=i+1)}
cp.opt2 = ifelse(i>1, sqrt(cart_max2$cptable[i,1]*cart_max2$cptable[i-1,1]),1)
cp.opt2
cart_opt2 = prune(cart_max2, cp = cp.opt2)
pred.cart_opt2 = ifelse(predict(cart_opt2, newdata = testset2)[,2]>0.5,1,0)
cm.c_opt2 = table(testset2$Heart_failure,pred.cart_opt2,dnn=c("actual","predict"))   
cm.c_opt2
round(prop.table(cm.c_opt2, margin = 1),4)

table[8,3] = (cm.c_opt2[1,1]+cm.c_opt2[2,2])/sum(cm.c_opt2)
table[8,4] = cm.c_opt2[1,2]/(cm.c_opt2[1,1]+cm.c_opt2[1,2])
table[8,5] = cm.c_opt2[2,1]/(cm.c_opt2[2,1]+cm.c_opt2[2,2])
paste("accuracy of CART is", (cm.c_opt2[1,1]+cm.c_opt2[2,2])/sum(cm.c_opt2))
paste("error rate (false positive) is", cm.c_opt2[1,2]/(cm.c_opt2[1,1]+cm.c_opt2[1,2]))
paste("error rate (false negative) is", cm.c_opt2[2,1]/(cm.c_opt2[2,1]+cm.c_opt2[2,2]))


# 4 most important features: random forest --------------------------------
rf2 = randomForest(Heart_failure ~ ., data = trainset2, importance = T, keep.inbag = T)
rf2
plot(rf2)
pred.rf2 = predict(rf2, newdata = testset2)
cm.r2 = table(testset2$Heart_failure,pred.rf2,dnn=c("actual","predict"))
cm.r2
round(prop.table(cm.r2, margin = 1),4)
table[9,3] = (cm.r2[1,1]+cm.r2[2,2])/sum(cm.r2)
table[9,4] = cm.r2[1,2]/(cm.r2[1,1]+cm.r2[1,2])
table[9,5] = cm.r2[2,1]/(cm.r2[2,1]+cm.r2[2,2])
paste("accuracy of random forest is",(cm.r2[1,1]+cm.r2[2,2])/sum(cm.r2))
paste("error rate (false positive) is", cm.r2[1,2]/(cm.r2[1,1]+cm.r2[1,2]))
paste("error rate (false negative) is", cm.r2[2,1]/(cm.r2[2,1]+cm.r2[2,2]))

table[,"Accuracy"] = round(as.numeric(table$Accuracy),digits = 4)
table[,"error.rate.false.positive."] = round(as.numeric(table$error.rate.false.positive.),digits = 4)
table[,"error.rate.false.negative."] = round(as.numeric(table$error.rate.false.negative.),digits = 4)
table


