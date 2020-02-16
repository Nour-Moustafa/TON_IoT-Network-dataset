library(h2o)
 # Starts H2O using localhost IP, port 54321, all CPUs, and 4g of memory
 h2o.init(ip = 'localhost', port = 54321)
 # open a web browser and wrtie http://localhost:54321/ 
 
 
#Pass training and testing sets to the machine learning models 
#training= filtered_train # 0r  
 training= orignal_train
 
# testing =filtered_test # or
 testing= orignal_test 
 
train.hex <- as.h2o(training)
test.hex <- as.h2o(testing)
 
label_no=ncol(training)           # the number of the  class/label cloumn
no_features=ncol(training)-1        # number of attributes/features selected 


train.hex[,label_no]= as.factor(train.hex[,label_no]) # label
test.hex[,label_no]= as.factor(test.hex[,label_no])


##########################################################
# machine Learning models

# 1) Gradient Boosting Machine(GBM)

 model.gbm <- h2o.gbm(y = label_no, x = 1:no_features, training_frame = train.hex, validation_frame=test.hex, 
                      ntrees = 15, max_depth = 5, min_rows =2,
                      learn_rate = 0.01, distribution= "multinomial",  nfolds=10,  model_id = "network_data_gbm_model")
 # retrieve the model performance
 perf.gbm <- h2o.performance(model.gbm, test.hex)
 perf.gbm 
 # retrieve the AUC for the performance object:
 h2o.auc(perf.gbm)
 h2o.F1(perf.gbm)
 plot(perf.gbm, type="roc")
 summary(model.gbm)
 # retrieve the gini value for both the training and validation/testing data:
 h2o.giniCoef(model.gbm, train=TRUE, valid=TRUE, xval=FALSE)
#gbm.test = h2o.predict(object =  model.gbm, newdata = test.hex)
 h2o.accuracy(perf.gbm)
 h2o.varimp_plot(model.gbm)
 ##################################
 # 3) Random Forest 
 
 model.rf <- h2o.randomForest(         
   y = label_no, x = 1:no_features, training_frame = train.hex, validation_frame=test.hex,              
   ntrees = 150,                  
   stopping_rounds = 2,          
   seed = 10000,nfolds=10,
   model_id = "network_data_RF_model")  
 
 perf.rf <- h2o.performance(model.rf, test.hex)
 h2o.auc(perf.rf)
 h2o.auc(model.rf, train=TRUE, valid=TRUE, xval=FALSE)
 plot(perf.rf, type="roc")
 
##############################
 # 3) Naive Bayes
 model.nb<- h2o.naiveBayes(y=label_no, x = 1:no_features, training_frame = train.hex, validation_frame=test.hex,nfolds=10,
                           laplace = 3, model_id = "network_data_nb_model")
 perf.nb <- h2o.performance(model.nb, test.hex)
 h2o.auc(perf.nb)
 h2o.auc(model.nb, train=TRUE, valid=TRUE, xval=FALSE)
 plot(perf.nb, type="roc") 
 
 ##################################
 # 4) Feed Fowrad deep learning 
 
 model.dl <- h2o.deeplearning(y = label_no, x = 1:no_features, training_frame = train.hex, validation_frame=test.hex, 
                              hidden=c(15,15),
                              epochs=10, nfolds=10,
                              model_id = "network_data_dl_model"
 )
 perf.dl <- h2o.performance(model.dl, test.hex)
 h2o.auc(perf.dl)
 h2o.auc(model.dl, train=TRUE, valid=TRUE, xval=FALSE)
 plot(perf.dl, type="roc")
 
 ########################
 
 #h2o.shutdown()
