library(tidyverse)
library(keras)

max_features = 1000

max_length = 100

batch_size = 32

#Read data from the csv files!
setwd("~/DATA HW 2")
train <- read.csv('train.csv',header = FALSE)
test <- read.csv('test.csv',header = FALSE)
#The third column of the dataset contains the news article text
text_train <- train[,3] 
text_test <- test[,3]
#one hot coding ...
y_train <- to_categorical(train[,1]-1)
y_test <- to_categorical(test[,1]-1)


#Output lengths of training and test data set
cat(length(text_train), 'train sequences\n')
#120000 train sequences
cat(length(text_test), 'test sequences\n')
#7600 test sequences

tokenizer = text_tokenizer(num_words = max_features)
tokenizer %>% 
  fit_text_tokenizer(text_train)

#removing words shorter than 4 letters
tokenizer$word_index<-tokenizer$word_index[nchar(names(tokenizer$word_index))>3]
tokenizer$word_index %>%
  head()

text_train <- texts_to_sequences(tokenizer, text_train)

#apply dictionary
text_test <- texts_to_sequences(tokenizer, text_test)

#zero padding
text_train <- text_train %>%
  pad_sequences(maxlen = max_length)
dim(text_train)

text_test <- text_test %>%
  pad_sequences(maxlen = max_length)
dim(text_test)



#Output dimensions of training and test inputs

cat('text_train shape:' ,  dim(text_train),'\n')

cat('text_test shape:' ,  dim(text_test),'\n')


#model specification
embedding_dims <- 32
num_units <- 64

model <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = max_length) %>%
  layer_dense(num_units) %>%
  bidirectional(layer_lstm(units = num_units)) %>%
  layer_dense(num_units) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(4) %>%
  layer_activation("softmax") %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )

summary(model)

#model fitting
batch_size <- 32
epochs <- 3

hist <- model %>%
  fit(
    text_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.3
  )

model%>% evaluate(text_test, y_test)

P = model%>%predict(text_test)
dim(model%>%predict(text_test))

l = rep(NA,7600)
for(i in 1:7600)
{
l[i] = (which.max(P[i,]))
}

hist(l )

#FYI, meaning of each label for Y:
#1: World
#2: Sports
#3: Business 
#4: Sci/Tech 