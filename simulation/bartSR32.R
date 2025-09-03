#install.packages("purrr")

options(java.parameters = c("-Xms1g", "-Xmx8g", "-XX:+UseG1GC"))
library(rJava)
rJava::.jinit()
rt <- rJava::.jcall("java/lang/Runtime", "Ljava/lang/Runtime;", "getRuntime")
rJava::.jcall(rt, "J", "maxMemory")/1024^3 
library(bartMachine)
library(tidyverse)
library(caret)
library(cvAUC)

X = read_csv("D:/WashU/PHD/Year 2/Sem 1/Research/Papers/Papers Simulation/INNERPaper/simulation/data/correctModelSR32_x.csv")
Y = read_csv("D:/WashU/PHD/Year 2/Sem 1/Research/Papers/Papers Simulation/INNERPaper/simulation/data/correctModelSR32_y.csv")

validation = function(X,Y,TestSize){
  TestID = sample(1:nrow(X),size = floor(nrow(X) *TestSize),replace = F)
  X_test = X[TestID,];X_train = X[-TestID,]
  Y_test = Y[TestID,];Y_train = Y[-TestID,]
  
  X_train = data.frame(X_train);X_test = data.frame(X_test);
  Y_train = data.frame(Y_train);Y_test = data.frame(Y_test);
  Y_train$x = factor(Y_train$x);Y_test$x = factor(Y_test$x);
  
  bart_machine = build_bart_machine(X_train,Y_train$x,num_trees = 50,k = 5,mem_cache_for_speed = FALSE)
  #post = mc.pbart(as.matrix(X_train), as.matrix(Y_train), as.matrix(X_test),nskip = 3000,ndpost = 1000,mc.cores=10)
  predProb = 1 - predict(bart_machine, X_test)
  pred = predict(bart_machine, X_test, type = "class")
  acc = mean(pred == Y_test$x)
  sens = sensitivity(pred,Y_test$x,positive = "1")
  spec = specificity(pred,Y_test$x,negative = "0")
  cstat = AUC(predProb, Y_test$x)
  balAcc = (sens+spec)/2
  results = list(acc,sens,spec,cstat,balAcc)
  names(results) = c("Accuracy","Sensitivity","Specificity","Cstat","BalanceAccuracy")
  return(results)
}
Times = 50
Accuracy = rep(0,Times)
Sens = rep(0,Times)
Spec = rep(0,Times)
cStat = rep(0,Times)
balAcc = rep(0,Times)
for (i in 1:Times) {
  res = validation(X,Y,0.2)
  Accuracy[i] = res$Accuracy
  Sens[i] = res$Sensitivity
  Spec[i] = res$Specificity
  cStat[i] = res$Cstat
  balAcc[i] = res$BalanceAccuracy
}
resultsAll = list(Accuracy,Sens,Spec,cStat,balAcc)
names(resultsAll) = c("Accuracy","Sensitivity","Specificity","Cstat","BalanceAccuracy")
save(resultsAll,file = "BART_sr32.RData")
