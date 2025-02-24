# Birmingham-Parking-Evaluation
Time series forecasting of car parking data in Birmingham, UK. Applied different forcasting models such as regression models, exponential smoothing, ARIMA, Neural Networks, and aggregated forecasting models. 


Initial Data Exploration
Loading in the data and doing some initial data evaluation:

library(forecast)
## Registered S3 method overwritten by 'quantmod':
##   method            from
##   as.zoo.data.frame zoo
library(readxl)
library(ggplot2)
library(zoo)
## 
## Attaching package: 'zoo'
## The following objects are masked from 'package:base':
## 
##     as.Date, as.Date.numeric
# Reading in the data
birmingham_parking_data <- read.csv("C:\\Users\\Graduate\\Documents\\ND\\TSF\\TSF_Project_data.csv")

# Calculate occupancy rate for ease of analysis
birmingham_parking_data$OccupancyRate <- round((birmingham_parking_data$Occupancy / birmingham_parking_data$Capacity) * 100, 2)

# Adjusting data types. Changed LastUpdated column to Date format
birmingham_parking_data$LastUpdated <- as.Date(birmingham_parking_data$LastUpdated, tryFormats = c("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"))

# Filter only for first parking lot system code to simplify our analysis
filtered_data <- birmingham_parking_data[birmingham_parking_data$SystemCodeNumber == "BHMBCCMKT01", ]

# Aggregate to get daily averages for parking rates. This was necessary since there were multiple data points for each day
daily_data <- aggregate(OccupancyRate ~ LastUpdated, data = filtered_data, FUN = mean, na.rm = TRUE)

# Take a look at the data
head(daily_data)
##   LastUpdated OccupancyRate
## 1  2016-10-04      31.55222
## 2  2016-10-05      22.86833
## 3  2016-10-06      26.09333
## 4  2016-10-07      26.92111
## 5  2016-10-08      46.18611
## 6  2016-10-09      22.12611
tail(daily_data)
##    LastUpdated OccupancyRate
## 68  2016-12-14      21.75471
## 69  2016-12-15      24.59944
## 70  2016-12-16      23.63778
## 71  2016-12-17      45.98611
## 72  2016-12-18      27.02667
## 73  2016-12-19      30.80111
Converting it to a time series object
# Calculated the number of days between the first and last data points in our data set. There are 76 days in our data set. 
# 10/04 is say 278 in the year and 12/19 is day 354 of the year.

park.ts <- ts(daily_data$OccupancyRate, start = c(2016, 1), end = c(2016, 76), freq = 76)
#park.ts <- ts(daily_data$OccupancyRate, start = c(2015, 278), end = c(2015, 354), freq = 365)

# Plot the data
autoplot(park.ts)


#autoplot(park.zoo) + 
#  scale_x_date(date_breaks = "1 week", date_labels = "%b %d") + labs(x = "Date", y = "OccupancyRate", title = "Parking Occupancy Rate Over Time") + theme_minimal()
Data Partition
nTotal <- 76 # Total number of observations
nTrain <- round(0.8 * nTotal) # 80% of data
nValid <- nTotal - nTrain # Remaining 20%

# Correctly partition the data
train.ts <- window(park.ts, start = c(2016, 1), end = c(2016, nTrain))
valid.ts <- window(park.ts, start = c(2016, nTrain+1), end = c(2016, nTrain+nValid))

# Plot to see how the valid.ts fits the train.ts
autoplot(train.ts) + 
  autolayer(valid.ts)


Training multiple forecasting models
Regression Model
library(forecast) # Loading in the necessary library

regression_model <- tslm(train.ts ~ trend)

# Create predictions
regression.forecast <- forecast(regression_model, h = nValid)

# Plot the model and compare with observed valid data
autoplot(regression.forecast) + 
  autolayer(valid.ts, series = "Observed") 


# Extract the residuals
checkresiduals(regression.forecast)


## 
##  Ljung-Box test
## 
## data:  Residuals from Linear regression model
## Q* = 25.083, df = 12, p-value = 0.01443
## 
## Model df: 0.   Total lags used: 12
# Calculate the accuracy of the model
accuracy_regression <- accuracy(regression.forecast, valid.ts)
accuracy_regression
##                         ME     RMSE      MAE        MPE     MAPE MASE
## Training set -1.746667e-16 9.353167 6.820075 -8.7072596 23.54615  NaN
## Test set      1.695683e+00 8.257515 5.456044  0.1181265 16.90603  NaN
##                     ACF1 Theil's U
## Training set  0.04982395        NA
## Test set     -0.02855185 0.8421669
Quadratic Regression Model
Decided to also run a regression model to capture quadratic trend. The MAPE values decreased slightly.

library(forecast) # Loading in the necessary library

regression_model_quad <- tslm(train.ts ~ trend + I(trend^2))

# Create predictions
regression.quad.forecast <- forecast(regression_model_quad, h= nValid, level = 0)

# Plot the model and compare with observed valid data
autoplot(regression.quad.forecast) + 
  autolayer(valid.ts, series = "Observed") +
  labs(title = "Forecasts from Quadratic Trend Regression") 


# Extract the residuals
checkresiduals(regression.quad.forecast)


## 
##  Ljung-Box test
## 
## data:  Residuals from Linear regression model
## Q* = 25.208, df = 12, p-value = 0.01387
## 
## Model df: 0.   Total lags used: 12
# Calculate the accuracy of the model
accuracy_lm_quad <- accuracy(regression.quad.forecast, valid.ts)
accuracy_lm_quad
##                         ME     RMSE      MAE       MPE     MAPE MASE
## Training set -1.746028e-16 9.329214 6.753309 -8.659746 23.32298  NaN
## Test set      4.477765e+00 9.277543 5.718846 10.532946 16.32828  NaN
##                     ACF1 Theil's U
## Training set  0.04588667        NA
## Test set     -0.02258417 0.9367115
ETS (Exponential Smoothing) Model:
# Using ZZZ because it will learn all these models, test all possibilities of additive or multiplicative error, trend and seasonality, and select the best model

park.ets <- ets(train.ts, model = "ZZZ") 
# Best model selected was M, N, A, meaning there's a multiplicative error, no trend, and an additive seasonality

# Create predictions
park.ets.forecast <- forecast(park.ets, h = nValid, level = 0)

# Plot the model
autoplot(park.ets.forecast) + 
  autolayer(valid.ts, series = "Observed") 


# Extract the residuals
checkresiduals(park.ets.forecast)


## 
##  Ljung-Box test
## 
## data:  Residuals from ETS(A,N,N)
## Q* = 24.618, df = 12, p-value = 0.01674
## 
## Model df: 0.   Total lags used: 12
# Calculate the accuracy of the model
accuracy_ets <- accuracy(park.ets.forecast, valid.ts)
accuracy_ets
##                        ME     RMSE      MAE       MPE     MAPE MASE        ACF1
## Training set -0.000763595 9.376672 6.905333 -8.800931 23.90818  NaN  0.05218597
## Test set      0.278442161 8.086801 5.840246 -5.202892 19.22501  NaN -0.02770765
##              Theil's U
## Training set        NA
## Test set     0.8244105
MAPE for the testing set it slightly higher than in the regression models. Lets try out some more models.

Holt-Winter’s Exponential Smoothing
hwin <- ets(train.ts, model = "MAA")

# Create predictions
hwin.pred <- forecast(hwin, h = nValid, level = 0)

# Plot model
autoplot(hwin.pred) + 
  autolayer(valid.ts, series = "Observed")


# Extract the residuals
checkresiduals(hwin.pred)


## 
##  Ljung-Box test
## 
## data:  Residuals from ETS(M,A,N)
## Q* = 24.875, df = 12, p-value = 0.01543
## 
## Model df: 0.   Total lags used: 12
# Calculate the accuracy of the model
accuracy_hwin <- accuracy(hwin.pred, valid.ts)
accuracy_hwin
##                      ME     RMSE      MAE         MPE     MAPE MASE        ACF1
## Training set -0.7462065 9.393907 7.135177 -11.5561096 25.27193  NaN  0.05305880
## Test set      1.8429612 8.290404 5.458908   0.6656856 16.83531  NaN -0.02874735
##              Theil's U
## Training set        NA
## Test set     0.8451406
ARIMA Model:
park.arima <- auto.arima(train.ts)

# Create predictions
park.arima.forecast <- forecast(park.arima, h = nValid) 

# Plot the model
autoplot(park.arima.forecast) + 
  autolayer(valid.ts, series = "Observed")


# Extract the residuals
checkresiduals(park.arima.forecast)


## 
##  Ljung-Box test
## 
## data:  Residuals from ARIMA(1,0,1) with non-zero mean
## Q* = 22.032, df = 10, p-value = 0.01494
## 
## Model df: 2.   Total lags used: 12
# Calculate the accuracy of the model
accuracy_arima <- accuracy(park.arima.forecast, valid.ts)
accuracy_arima
##                       ME     RMSE      MAE       MPE     MAPE MASE        ACF1
## Training set -0.01757687 8.893316 6.507833 -8.006349 22.43309  NaN -0.06569886
## Test set      0.48146173 8.474017 5.899766 -4.479616 19.05320  NaN -0.13038669
##              Theil's U
## Training set        NA
## Test set     0.8732658
Neural Network (NN)
set.seed(123)
p <- 5 # Number of previous time steps used for forecast === 6
P <- 1 # Number of previous seasonal values to use  last wk ====1
size <- 4 # Number of hidden nodes 
# ^ chose these values randomly

# repeats - number of iterations or epochs to train the neural network
park.nnetar <- nnetar(train.ts, repeats =20, p = p, P = P, size = size) ## NN model
## Warning in nnetar(train.ts, repeats = 20, p = p, P = P, size = size): Series
## too short for seasonal lags
park.nnetar.forecast <- forecast(park.nnetar, h = nValid)

autoplot(park.nnetar.forecast) +
  autolayer(valid.ts, series = "Observed") 


# Calculate the accuracy
accuracy(park.nnetar.forecast, valid.ts)["Test set","MAPE"]
## [1] 14.81979
Aggregating Multiple Forecasts:
Lets investigate if we aggregate some models if that helps improves the accuracy scores.

Simple Average:
num.models <- 6

park.comb.simple.avg <- (regression.forecast$mean + park.ets.forecast$mean + 
                          park.arima.forecast$mean + hwin.pred$mean + 
                           regression.quad.forecast$mean + park.nnetar.forecast$mean) /num.models #adding them all up / # of models

# Plot the model
autoplot(train.ts) +
  autolayer(park.comb.simple.avg, series = "Simple Avg Comb") +
  autolayer(valid.ts, series = "Observed")+
  labs(title = "Simple Average vs. Observed") 


# Extract the residuals
checkresiduals(park.comb.simple.avg)


## 
##  Ljung-Box test
## 
## data:  Residuals
## Q* = 0.62529, df = 3, p-value = 0.8906
## 
## Model df: 0.   Total lags used: 3
# Calculate the accuracy
accuracy_simple_avg <- accuracy(park.comb.simple.avg, valid.ts)
accuracy_simple_avg
##                ME     RMSE      MAE        MPE     MAPE         ACF1 Theil's U
## Test set 1.415712 7.129366 5.065732 -0.1047628 16.01854 -0.009528994 0.7242278
Trimmed Mean
# Collect the forecasts in a data frame
forecast.vectors.df <- data.frame(cbind(regression.forecast$mean, park.ets.forecast$mean, park.arima.forecast$mean, hwin.pred$mean, regression.quad.forecast$mean, park.nnetar.forecast$mean))

# Function to compute trimmed mean for each row separately - Using 20% trimming
# That is, we are trimming one model above and one model below as we have 6 models. 
forecast.vectors.df$comb.trimmed.avg <- apply(forecast.vectors.df, 1, function(x) mean(x, trim = 0.2))

# Convert the object into a ts object
# Using 60.8 as that is the day we start our validation set
park.comb.trimmed.avg <- ts(forecast.vectors.df$comb.trimmed.avg, start = c(2016, 60.8), frequency = 76)

# Plot model
autoplot(train.ts) +
  autolayer(park.comb.trimmed.avg, series = "Trimmed Avg Comb") +
  autolayer(valid.ts, series = "Observed")+
  labs(title = "Trimmed Mean vs. Observed") 


# Extract the residuals
checkresiduals(park.comb.trimmed.avg)


## 
##  Ljung-Box test
## 
## data:  Residuals
## Q* = 0.98647, df = 3, p-value = 0.8045
## 
## Model df: 0.   Total lags used: 3
# Calculate the accuracy
accuracy_trimmed_avg <- accuracy(park.comb.trimmed.avg, valid.ts)
accuracy_trimmed_avg
##                ME     RMSE      MAE     MPE     MAPE       ACF1 Theil's U
## Test set 2.246734 9.040512 6.046171 1.25801 18.25047 -0.0216057 0.8825417
Model Evaluation
Compare accuracy of all models, focusing on MAPE:

all_accuracy <- round(c(
  LM = accuracy(regression.forecast, valid.ts)["Test set","MAPE"], 
  LM_QUAD = accuracy(regression.quad.forecast, valid.ts)["Test set","MAPE"],
  HWIN = accuracy(hwin.pred, valid.ts)["Test set","MAPE"],
  ETS = accuracy(park.ets.forecast, valid.ts)["Test set","MAPE"],
  ARIMA = accuracy(park.arima.forecast, valid.ts)["Test set","MAPE"],
  NN = accuracy(park.nnetar.forecast, valid.ts)["Test set","MAPE"],
  comb.simple.avg = accuracy(park.comb.simple.avg, valid.ts)["Test set","MAPE"],
  comb.reg = accuracy(park.comb.trimmed.avg, valid.ts)["Test set","MAPE"]), 2)

#sort(all_accuracy)

# Simply converting it to a data frame so we can visualize more easily:
accuracy_df <- data.frame(Model = names(all_accuracy), MAPE = all_accuracy)

# Sort the data frame by MAPE values
accuracy_df <- accuracy_df[order(accuracy_df$MAPE), ]

print(accuracy_df)
##                           Model  MAPE
## NN                           NN 14.82
## comb.simple.avg comb.simple.avg 16.02
## LM_QUAD                 LM_QUAD 16.33
## HWIN                       HWIN 16.84
## LM                           LM 16.91
## comb.reg               comb.reg 18.25
## ARIMA                     ARIMA 19.05
## ETS                         ETS 19.23
