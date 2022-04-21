
# The ultimate match: ARMA vs recurrent NN
# we want to test if we can predict better with ARMA or a recurrent NN

rm(list=ls())

# in order to test this, we use a famous data set in Machine Learning: the Jena data set
# the Weather Station at the Max-Planck-Institute for Biogeochemistry in Jena, Germany: http://www.bgc-jena.mpg.de/wetter/.
# In this dataset, fourteen different quantities (such air temperature, atmospheric pressure, humidity, wind direction, etc.) 
# are recorded every ten minutes, over several years. The original data goes back to 2003, but we limit ourselves to data from 2009-2016. 
# actually I am only going to use 14 days as I have a small computer

# I am going to focus on atmospheric pressure, but you can use similar considerations for other variables

# you can download the data with the following commented code



library(tibble)
library(readr)
library(ggplot2)
library(keras)


data=read_csv("jena_climate_2009_2016.csv")

# which variable? We consider pressure (variable 2)
data=data.matrix(data[,2])

# For every day there are day_count observations
day_count=6*24

# The amount of data is very large, we consider only 14 days
day_max=100
n_obs=day_count*day_max
data=data[1:n_obs]

# # Out of the 100 days, I will use 50 for training, 30 for validation, hence 20 for test
day_train=50
day_validate=30

train_data <- data[1:(day_train*day_count)]
validation_data <- data[(day_train*day_count+1):((day_train+day_validate)*day_count)]
test_data <- data[((day_train+day_validate)*day_count+1):(day_max*day_count)]


# We normalize by computing the mean and sd from the data
# this preprocessing step shoud always be done in NN

data.mean=mean(train_data)
data.sd=sd(train_data)
train_data=scale(train_data, center = data.mean, scale = data.sd)
validation_data=scale(validation_data, center = data.mean, scale = data.sd)
test_data=scale(test_data, center = data.mean, scale = data.sd)

n_train=length(train_data)
n_val=length(validation_data)
n_test=length(test_data)

# let's plot training, validation and test data
df.train=data.frame(X=1:n_train,Y=train_data)
df.valid=data.frame(X=(n_train+1):(n_train+n_val),Y=validation_data)
df.test=data.frame(X=(n_train+n_val+1):n_obs,Y=test_data)

gp=ggplot()+geom_line(data=df.train,aes(X,Y),color="red")+geom_line(data=df.valid,aes(X,Y),color="black")+xlab("Time")+ylab("Pressure (normalized)")
gp+geom_line(data=df.test,aes(X,Y),color="blue")





# shall we try a small modification? Just delete the first "step" entries
data_time_format <- function(data,step){
  n=length(data)
  seq=(step+1):n
  n_form=length(seq)
  y=data[(step+1):n]
  X=matrix(NaN,nrow=n_form,ncol=step)
  
  for(i in (step+1):n){
    X[i-step,]=data[(i-1):(i-step)]
  }
  
  return(list(X,y))
}

# now we write training, validation and test set in input-output format
step=3
c(X.train,y.train) %<-% data_time_format(train_data,step)
c(X.valid,y.valid) %<-% data_time_format(validation_data,step)
c(X.test,y.test) %<-% data_time_format(test_data,step)

# your input has to be in this form [number of replicates,length of time series,lags]
# your output has to be in this form [number of replicates,length of time series]

dim(y.train)=c(1,length(y.train))
dim(y.valid)=c(1,length(y.valid))
dim(y.test)=c(1,length(y.test))


data.training=array(NA,dim=c(1,nrow(X.train),step))
data.val=array(NA,dim=c(1,nrow(X.valid),step))
data.test=array(NA,dim=c(1,nrow(X.test),step))

#creating input variables for training data ... note that the first dimension has to be the number of replicates ... here it's 1 b/c there's only one series. 
data.training[1,,]=X.train

#creating input variables for validation data
data.val[1,,]=X.valid

#creating input variables for test data
data.test[1,,]=X.test

# let's start building the model

# setting the same random seed
# use_session_with_seed(1)

inputs = layer_input(shape=list(NULL,dim(data.training)[3]))

lstm <- inputs %>% 
  layer_dense(units = 32, activation = 'tanh') %>% 
  layer_lstm(units=dim(data.training)[3],  return_sequences=TRUE) %>% 
  layer_dense(units = 32, activation = 'tanh') %>% 
  time_distributed(layer_dense(units = 1,activation = 'linear'))


model <- keras_model(inputs=inputs,outputs=lstm)


summary(model)


callbacks=list(callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 30,
                                       verbose = 0, mode = "auto"),
               callback_model_checkpoint(filepath='best_model.hdf5',save_freq='epoch',save_best_only = TRUE)
)


#model fitting
model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error',
  metrics = 'mse'
)

init.time=proc.time()
model %>% fit(
  data.training,
  y.train,
  epochs=1000,
  verbose=1,
  validation_data=list(data.val,y.valid),
  validation_steps=1,
  callbacks = callbacks
)
proc.time()-init.time

model_best = load_model_hdf5("best_model.hdf5")


# we predict validation and test data togheter
y.val.test.or=c(validation_data,test_data)
c(X.val.test,y.val.test) %<-% data_time_format(y.val.test.or,step)

data.val.test=array(NA,dim=c(1,nrow(X.val.test),step))
data.val.test[1,,]=X.val.test
pred.val.test  =  model_best %>% predict(data.val.test)



# let's see how the test data perform
x.vis=(n_val+1):(n_val+n_test-step)
RMSE.short.NN=sum((pred.val.test[x.vis]-test_data[1:(n_test-step)])^2)
df=data.frame(X=1:(n_val+n_test-step),Y1=y.val.test*data.sd+data.mean,Y2=c(pred.val.test)*data.sd+data.mean)
gp.val=ggplot()+geom_line(data=df,aes(X,Y1),color="red")+geom_line(data=df,aes(X,Y2),color="black")+ylab("Pressure (mbar)")+xlab("Time")+xlim(x.vis[1],tail(x.vis, n=1))+ylim(min(y.val.test[x.vis]*data.sd+data.mean),max(y.val.test[x.vis]*data.sd+data.mean))
gp.val
# it seems to work quite well


####################################################################
############ short term forecasting with ARMA
####################################################################

# forecast point by point


library(forecast)

short.term.forc.ARMA <- function(data.train,data.test,pred.hor){
  
  h=length(data.test)
  pred=matrix(NaN,nrow=h,ncol=pred.hor)
  for (i in 1:h){
    # notice I am also allowing model selection for d
    mod.arma=auto.arima(data.train,ic="bic")
    forc=forecast(mod.arma,h=pred.hor)
    pred[i,]=as.numeric(forc$mean)
    data.train=c(data.train,data.test[i])
  }
  return(pred)
}


data.arma=ts(data[1:((day_train+day_validate)*day_count)])

pred.hor=1
short.term.pred.ARMA=short.term.forc.ARMA(data.arma,test_data,pred.hor)

RMSE.short.ARMA=sum((short.term.pred.ARMA[1:(n_test-step),1]-test_data[1:(n_test-step)])^2)

#forc.mean=as.numeric(short.term.pred.ARMA*data.sd+data.mean)

df2=data.frame(X=(n_val+1):(n_val+n_test),Y=as.numeric(short.term.pred.ARMA[,1]*data.sd+data.mean))
gp.val=gp.val+geom_line(data=df2,aes(X,Y),color="blue")#+ylim(min(forc.mean),max(y.val.test[x.vis]*data.sd+data.mean))
gp.val

c(RMSE.short.NN,RMSE.short.ARMA)

save('Results.Data')
