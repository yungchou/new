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


splitting <- function(df, training=0.6, testing=0.2, holding=0.2, logging=FALSE){

  if (logging) cat('\nEntering the function, splitting')

if( (training+testing+holding)!=1 ){
  training = 0.6 ; testgin = 0.2 ; holding = 0.2
  cat('Sum of training, testing and holding portions not equal to 1,\n',
      'Using training = 0.6, testgin = 0.2, and holding = 0.2, instead.')
}

#----------------
# Splitting data
#----------------

if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
set.seed(0-0)

hold.data <- sample_frac( df, holding )

# Excluding the hold data from the imported dataset
temp <- setdiff(df, hold.data)

part <- sample(2 , nrow(temp) ,replace=TRUE ,prob=c(
  training/(training+testing),
  testing/(training+testing)
  ))

train.data <- temp[part==1,]
test.data  <- temp[part==2,]

cat('\nImported dataset =',nrow(df),'rows',
    '\nTrain portion (',training,') =',nrow(train.data),
    '\nTest  portion (',testing ,') =',nrow(test.data),
    '\nHold  portion (',holding ,') =',nrow(hold.data))

  if (logging) cat('\nLeaving the function, splitting')

return (train.data, test.data, hold.data)

}