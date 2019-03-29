if (!require('caret'  )) install.packages('caret', dependencies=c("Depends","Suggests") ); library(caret  )
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)

df <- read.csv( mydataset <- './dataset/creditcard.csv' )

source("./helper/yc.helper.R");
missingness(df)
splitting(df, 0.8) # portion of data for training

