# Birmingham-Parking-Evaluation
Time series forecasting of car parking data in Birmingham, UK. Applied different forcasting models such as regression models, exponential smoothing, ARIMA, Neural Networks, and aggregated forecasting models. 

---
title: "Birmingham Parking Evaluation"
author: "Olivia Marcinkus, Ruthie Montella, and Giulia Neves Monteiro"
date: "2025-02-12"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Initial Data Exploration

Loading in the data and doing some initial data evaluation:
```{r}
library(forecast)
library(readxl)
library(ggplot2)
library(zoo)

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
tail(daily_data)
```

## Converting it to a time series object
```{r}
# Calculated the number of days between the first and last data points in our data set. There are 76 days in our data set. 
# 10/04 is say 278 in the year and 12/19 is day 354 of the year.

park.ts <- ts(daily_data$OccupancyRate, start = c(2016, 1), end = c(2016, 76), freq = 76)
#park.ts <- ts(daily_data$OccupancyRate, start = c(2015, 278), end = c(2015, 354), freq = 365)

# Plot the data
autoplot(park.ts)

#autoplot(park.zoo) + 
#  scale_x_date(date_breaks = "1 week", date_labels = "%b %d") + labs(x = "Date", y = "OccupancyRate", title = "Parking Occupancy Rate Over Time") + theme_minimal()
```

## Data Partition
```{r}
nTotal <- 76 # Total number of observations
nTrain <- round(0.8 * nTotal) # 80% of data
nValid <- nTotal - nTrain # Remaining 20%

# Correctly partition the data
train.ts <- window(park.ts, start = c(2016, 1), end = c(2016, nTrain))
valid.ts <- window(park.ts, start = c(2016, nTrain+1), end = c(2016, nTrain+nValid))

# Plot to see how the valid.ts fits the train.ts
autoplot(train.ts) + 
  autolayer(valid.ts)

```

## Training multiple forecasting models

### Regression Model
```{r}
library(forecast) # Loading in the necessary library

regression_model <- tslm(train.ts ~ trend)

# Create predictions
regression.forecast <- forecast(regression_model, h = nValid)

# Plot the model and compare with observed valid data
autoplot(regression.forecast) + 
  autolayer(valid.ts, series = "Observed") 

# Extract the residuals
checkresiduals(regression.forecast)

# Calculate the accuracy of the model
accuracy_regression <- accuracy(regression.forecast, valid.ts)
accuracy_regression
```

### Quadratic Regression Model 

Decided to also run a regression model to capture quadratic trend. The MAPE values decreased slightly.
```{r}
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

# Calculate the accuracy of the model
accuracy_lm_quad <- accuracy(regression.quad.forecast, valid.ts)
accuracy_lm_quad
```


### ETS (Exponential Smoothing) Model: 

```{r}
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

# Calculate the accuracy of the model
accuracy_ets <- accuracy(park.ets.forecast, valid.ts)
accuracy_ets
```
MAPE for the testing set it slightly higher than in the regression models. Lets try out some more models.


### Holt-Winter's Exponential Smoothing
```{r}
hwin <- ets(train.ts, model = "MAA")

# Create predictions
hwin.pred <- forecast(hwin, h = nValid, level = 0)

# Plot model
autoplot(hwin.pred) + 
  autolayer(valid.ts, series = "Observed")

# Extract the residuals
checkresiduals(hwin.pred)

# Calculate the accuracy of the model
accuracy_hwin <- accuracy(hwin.pred, valid.ts)
accuracy_hwin
```

### ARIMA Model: 
```{r}
park.arima <- auto.arima(train.ts)

# Create predictions
park.arima.forecast <- forecast(park.arima, h = nValid) 

# Plot the model
autoplot(park.arima.forecast) + 
  autolayer(valid.ts, series = "Observed")

# Extract the residuals
checkresiduals(park.arima.forecast)

# Calculate the accuracy of the model
accuracy_arima <- accuracy(park.arima.forecast, valid.ts)
accuracy_arima
```

### Neural Network (NN)
```{r}
set.seed(123)
p <- 5 # Number of previous time steps used for forecast === 6
P <- 1 # Number of previous seasonal values to use  last wk ====1
size <- 4 # Number of hidden nodes 
# ^ chose these values randomly

# repeats - number of iterations or epochs to train the neural network
park.nnetar <- nnetar(train.ts, repeats =20, p = p, P = P, size = size) ## NN model

park.nnetar.forecast <- forecast(park.nnetar, h = nValid)

autoplot(park.nnetar.forecast) +
  autolayer(valid.ts, series = "Observed") 

# Calculate the accuracy
accuracy(park.nnetar.forecast, valid.ts)["Test set","MAPE"]

```


### Aggregating Multiple Forecasts:

Lets investigate if we aggregate some models if that helps improves the accuracy scores.

#### Simple Average:
```{r}
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

# Calculate the accuracy
accuracy_simple_avg <- accuracy(park.comb.simple.avg, valid.ts)
accuracy_simple_avg
```

#### Trimmed Mean
```{r}
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

# Calculate the accuracy
accuracy_trimmed_avg <- accuracy(park.comb.trimmed.avg, valid.ts)
accuracy_trimmed_avg
```


# Model Evaluation

Compare accuracy of all models, focusing on MAPE: 
```{r}
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
```
