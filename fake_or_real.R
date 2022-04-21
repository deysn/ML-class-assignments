rm(list=ls())

library(tidyverse)
library(keras)

#data import
dataset <- read_csv('fake_or_real_news.csv')

#splitting training and test datasets
set.seed(2020)
index.train <- sample(1:nrow(dataset),round(0.8*nrow(dataset))) 

dataset_train <- dataset[index.train,]
y_train <- as.numeric(dataset_train[,4]=='FAKE')
text_train <- dataset_train$text

dataset_test <- dataset[-index.train,]
y_test <- as.numeric(dataset_test[,4]=='FAKE')
text_test <- dataset_test$text

#remove special characters
text_train <- str_replace_all(text_train, "[[:punct:]]", " ")
text_test <- str_replace_all(text_test, "[[:punct:]]", " ")

#tokenizer: creating dictionary
max_features <- 1000
tokenizer <- text_tokenizer(num_words = max_features)

tokenizer %>% 
  fit_text_tokenizer(text_train)

#removing words shorter than 4 letters
tokenizer$word_index<-tokenizer$word_index[nchar(names(tokenizer$word_index))>3]
tokenizer$word_index %>%
  head()

#apply dictionary
text_seqs_train <- texts_to_sequences(tokenizer, text_train)

#apply dictionary
text_seqs_test <- texts_to_sequences(tokenizer, text_test)

#zero padding
maxlen <- 100
x_train <- text_seqs_train %>%
  pad_sequences(maxlen = maxlen)
dim(x_train)

x_test <- text_seqs_test %>%
  pad_sequences(maxlen = maxlen)
dim(x_test)


#model specification
embedding_dims <- 32
num_units <- 64

model <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dense(num_units) %>%
  bidirectional(layer_lstm(units = num_units)) %>%
  layer_dense(num_units) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid") %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

summary(model)

#model fitting
batch_size <- 32
epochs <- 3

hist <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.3
  )

model %>% evaluate(x_test, y_test)


