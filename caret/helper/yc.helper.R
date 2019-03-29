missingness <- function(df, logging=FALSE){

  if (logging) cat('\nEntering the function, missingness\n')

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

    # Handle missing data here

  } else {
    cat('The imported dataset has no missing data.')
  }

  if (logging) cat('\nLeaving the function, missingness\n')

}


splitting <- function(df, trian.portion=0.7, loggin=FALSE){

  if (logging) cat('\nEntering the function, splitting')

#----------------
# Splitting data
#----------------

if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
set.seed(0-0)

# Rows excluded fomr the imported dataset and reserved for 
# later scoring the model with predictions
hold.data <- sample_frac( df, 1-train.portion )

# Spliting the rest for 80% training and 20% testing
temp <- setdiff(df, hold.data)
part <- sample(2 , nrow(temp) ,replace=TRUE ,prob=c(
  0.8, # train data
  0.2  # test data
  ))

train.data <- temp[part==1,]  # original training data
test.data  <- temp[part==2,]

cat('\nImported dataset =',nrow(df),'rows',
    '\nSplit =',info['split'],
    '\nTrain data  =',nrow(train.data),
    '\nTest data   =',nrow(test.data),
    '\nHold data   =',nrow(hold.data))

  if (logging) cat('\nLeaving the function, splitting')

}