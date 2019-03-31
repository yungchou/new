if (!require('caret')) install.packages('caret', dependencies=c("Depends","Suggests") ); library(caret  )
if (!require('GGally')) install.packages('GGally'); library(GGally)
if (!require('plotly')) install.packages('plotly'); library(plotly)
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)

system.time(df1 <- data.table::fread( mydataset <- './dataset/creditcard.csv' ))

str(df)
ggcorr(df[1:100,2:8], palette = "RdBu", label = TRUE)
ggplotly(ggpairs(df[1:100,2:8]))

source("./helper/yc.helper.R");
missingness(df)
mydf <- splitting(df, training=0.6, testing=0.2, holding=0.2)

