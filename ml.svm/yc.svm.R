# Yung Chou's Demo on SVM
{
  #-----------------
  # CUSTOM SETTINGS
  #-----------------
  info <- c()
  info['split'] <- 0.7  # % of dataset as training data
  saveRDS(info,'info.rds')
}

#---------
# Dataset
#---------
df <- read.csv( mydataset <- './dataset/creditcard.csv' )

#-------------
# Missingness
#-------------
if( any(is.na(df)) ){
  if (!require('naniar')) install.packages('naniar'); library(naniar)
  jpeg('missingness.jpg')
  gg_miss_upset(df)
  dev.off()

  #------------------------------
  # Percentage of missing values
  #------------------------------
  missing <- function(x) { round(( sum(is.na(x))/length(x) )*100, 2) }
  apply(df,2,missing)

  # Do somehting about missing data here
} else {
  cat('The imported dataset,',mydataset,', has no missing data.')
}

#----------------
# Splitting data
#----------------
#df <- df[1:1000,]

if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
set.seed(0-0)

# Rows excluded fomr the imported dataset and
# reserved for later scoring the model with predictions
hold.data <- sample_frac( df, 1-info['split'] ) ;head(hold.data)

# Spliting the rest for training and testing
temp <- setdiff(df, hold.data)
part <- sample(2 , nrow(temp) ,replace=TRUE ,prob=c(
  0.8, # train data
  0.2  # test data
  ))

train.data <- temp[part==1,]  # original training data
test.data  <- temp[part==2,]

cat('Imported dataset =',nrow(df),'rows',
    '\nSplit =',info['split'],
    '\nTrain data  =',nrow(train.data),
    '\nTest data   =',nrow(test.data),
    '\nHold data   =',nrow(hold.data))

#-----
# SVM
#-----

# Ref: H2O has not implemented SVM. However Sparkling Water exposes
#      Spark's SVM implementation. (http://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-support-vector-machines-svms).

if (!require('e1071')) install.packages('e1071'); library(e1071)

svm.model <- svm(train.data$Class~., train.data, cross=5)

summary(svm.model)
