#!/usr/bin/env python
# coding: utf-8

# In[94]:


#import the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams


# In[95]:


#load the dataset
df = pd.read_csv('datasets_302666_618181_AAPL.csv')
df.head()


# In[96]:


#check the shape
df.shape


# In[97]:


#We will consider the volume of stocks as the depedent variable


# In[98]:


df_vol = df[['Date', 'Volume']]
df_vol.head()


# In[99]:


#set date as the index column for doing time series analysisdf.set_index(0)
df_vol.dtypes


# In[100]:


#We can observe that date is represented as object. We need to convert this to timestamp
df_vol['Date'] = df_vol['Date'].astype('datetime64')


# In[101]:


df_vol.dtypes


# In[102]:


#We have converted the date to a datetime64 data type.


# In[103]:


#check the first five rows
df_vol.head()


# In[104]:


#check the last five rows
df_vol.tail()


# In[105]:


#check for missing values
df_vol.isnull().sum()


# In[106]:


#check the summary statistics
df_vol.describe()


# In[107]:


#We have no missing values in the data


# In[108]:


#set the index column as the Date column
df_final = df_vol.set_index('Date')
df_final.head()


# In[109]:


#plot the time series
rcParams['figure.figsize'] = 12,6
df_final.plot()


# Looking at the above figure, it looks like the series is stationary.

# In[110]:


#Let us check this assumption statistically using the Dickey Fuller test at alpha = 0.05


# In[111]:


#Import the dickey fuller test
from statsmodels.tsa.stattools import adfuller

#formulate the null and alternate hypothesis
#H0 : Series is not stationary
#H1 : Series is stationary

#run the test
adfuller(df_final)


# As the p value here (0.018) is less than 0.05, we reject the H0 that the time series is not stationary.
# 
# Therefore, we can conclude that the series is stationary at alpha = 0.05

# In[112]:


#plot acf and pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[113]:


#plot acf
plt.figure(figsize=(12,8))
plot_acf(df_final,lags=50,  ax=plt.gca())
plt.show()


# In[114]:


#plot pacf
plt.figure(figsize=(12,8))
plot_pacf(df_final, lags=50, ax=plt.gca())
plt.show()


# In[115]:


#Seasonality is observed after every 5 days is visible


# In[116]:


#Check for seasonality, trend, errors by decomposing the data


# In[117]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[118]:


decomposition = seasonal_decompose(df_final, model='additive', period=5)
decomposition.plot()
plt.show()


# In[119]:


decomposition = seasonal_decompose(df_final, model='multiplicative', period=5)
decomposition.plot()
plt.show()


# If the magnitude of the seasonal component changes with time, then the series is multiplicative. Otherwise, the series is additive.
# 
# Here, the series is additive based on above plots.

# ### Train Test Split 

# In[120]:


#Before building the model, we will split the data into train and test set in the ratio 80:20


# In[153]:


train = df[pd.to_datetime(df['Date']) < pd.to_datetime('2019-06-14')]
train.shape


# In[154]:


test = df[pd.to_datetime(df['Date']) >= pd.to_datetime('2019-06-14')]
test.shape


# In[155]:


train_vol = train[['Volume']]
test_vol = test[['Volume']]


# In[156]:


train_vol.head()


# In[157]:


test_vol.head()


# ### SARIMA Model

# Since the data contains seasonality, we cannot use the ARIMA model.
# 
# Therefore, here we use the **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** model 

# In[129]:


#calculate the values of p,d,q

import itertools

p=q=range(0,3)
#no differencing so d is 0
d=range(0,1)
pdq = list(itertools.product(p,d,q))

model_pdq = [(x[0], x[1], x[2], 5) for x in list(itertools.product(p,d,q))]


# In[130]:


df_sarima = pd.DataFrame(columns=['param','seasonal','AIC'])
df_sarima


# In[158]:


#import statsmodels
import statsmodels.api as sm

for param in pdq:
    for param_seasonal in model_pdq:
        mod = sm.tsa.statespace.SARIMAX(train_vol, 
                                        order=param, 
                                        seasonal_order=param_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        
        results_SARIMA = mod.fit()
        print('SARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal, results_SARIMA.aic))
        df_sarima = df_sarima.append({'param':param,'seasonal':param_seasonal ,'AIC': results_SARIMA.aic}, ignore_index=True)


# In[159]:


df_sarima.sort_values(by=['AIC']).head(5)


# In[160]:


##SARIMA(1,0,2)(0,0,2,5) - AIC(6637.695675)
mod = sm.tsa.statespace.SARIMAX(train_vol, 
                                order=(1,0,2), 
                                seasonal_order=(0,0,2,5),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary())


# In[134]:


results.plot_diagnostics(figsize=(15,8))
plt.show()


# In[161]:


#predict on test data
pred_SARIMA = results.get_forecast(steps=len(test_vol))
pred_SARIMA.predicted_mean


# In[163]:


#evaluate the model
from math import sqrt
from sklearn.metrics import mean_squared_error

rmse = sqrt(mean_squared_error(test_vol, pred_SARIMA.predicted_mean))
print(rmse)


# In[164]:


#tabulate the results into a pandas dataframe
resultsDf = pd.DataFrame({'Test RMSE': rmse},
                        index=['SARIMA (1, 0, 2)(0, 0, 2,5)'])
resultsDf


# In[165]:


#plot the time series for SARIMA model
plt.plot(train_vol,label='Training Data')
plt.plot(test_vol,label='Test Data')
plt.plot(test_vol.index, pred_SARIMA.predicted_mean, label='Predicted Data - SARIMA')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[139]:


#The forecast is not as per the terms we want. Let's build the SARIMAX model.


# ### SARIMAX Model

# In[167]:


#create the exogeneous variables

ex_train = train[['High','Low','Open','Close','Adj Close']]
ex_test = test[['High','Low','Open','Close','Adj Close']]


# In[168]:


ex_train.head()


# In[169]:


ex_test.head()


# In[170]:


df_sarimax = pd.DataFrame(columns=['param','seasonal', 'AIC'])
df_sarimax


# In[171]:


#use SARIMAX
for param in pdq:
    for param_seasonal in model_pdq:
        mod = sm.tsa.statespace.SARIMAX(train['Volume'],
                                        exog=ex_train,
                                        order=param,
                                        seasonal_order=param_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
            
        results_SARIMAX = mod.fit()
        print('SARIMAX{}{} - AIC:{}'.format(param, param_seasonal, results_SARIMAX.aic))
        df_sarimax = df_sarimax.append({'param':param,'seasonal':param_seasonal ,'AIC': results_SARIMAX.aic}, ignore_index=True)


# In[173]:


df_sarimax.sort_values(by=['AIC']).head(5)


# In[174]:


## SARIMAX(1, 0, 2)(0, 0, 2, 5)

mod = sm.tsa.statespace.SARIMAX(train['Volume'],
                                exog=ex_train,
                                order=(1,0,2),
                                seasonal_order=(0,0,2,5),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())


# In[179]:


#predict on the test set
pred_SARIMAX = results.get_forecast(steps=len(test),exog=ex_test)
pred_SARIMAX.predicted_mean


# In[176]:


rmse = sqrt(mean_squared_error(test.Volume,pred_SARIMAX.predicted_mean))
print(rmse)


# In[177]:


#tabulate the results into a pandas dataframe
resultsDf_temp = pd.DataFrame({'Test RMSE': rmse},
                        index=['SARIMAX (1, 0, 2)(0, 0, 2, 5)'])

resultsDf = pd.concat([resultsDf, resultsDf_temp])
resultsDf


# In[183]:


#plot the time series for SARIMAX model
plt.plot(train_vol,label='Training Data')
plt.plot(test_vol,label='Test Data')
plt.plot(test_vol.index, pred_SARIMA.predicted_mean, label='Predicted Data - SARIMA')
plt.plot(test_vol.index, pred_SARIMAX.predicted_mean, label='Predicted Data - SARIMAX')
plt.legend(loc='best')
plt.grid()
plt.show()


# We can observe that as compared to the SARIMA model, the SARIMAX model has a lower rmse value and better fit.

# ### Time Series using Facebook Prophet library

# In[191]:


#import the required libraries
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())


# In[188]:


#create dataframe with required variables
df_prophet = pd.DataFrame()
#prophet requires us to set the datetime column as ds and dependent variable as y
df_prophet['ds'] = pd.to_datetime(df['Date'])
df_prophet['y'] = df['Volume']
df_prophet['High']= df['High']
df_prophet['Low']=df['Low']

df_prophet.head()


# In[195]:


#initialize the model
prophet=Prophet(seasonality_mode='additive', weekly_seasonality=True)
#add the country holidays to account for unnatural events
prophet.add_country_holidays(country_name='US')
#fit the model on the data
prophet.fit(df_prophet[df_prophet['ds'] <= pd.to_datetime('2019-06-13')])
#create future dataframe for next 43 days
future = prophet.make_future_dataframe(periods=43, freq=us_bd)
#make the predictions
forecast = prophet.predict(future)

#plot the predictions
fig = prophet.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), prophet, forecast)
plt.show()
fig2 = prophet.plot_components(forecast)
plt.show()


# Prophet has included the original data as the black dots and the blue line is the forecast model.
# 
# The light blue area is the confidence interval. Using the add_changepoints_to_plot function added the red lines; the vertical dashed lines are changepoints Prophet identified where the trend changed, and the solid red line is the trend with all seasonality removed. 
# 
# 
# The components plot consists of 3 sections: the trend, the holidays, and the seasonality. The sum of those 3 components account for the entirety of the model in fact. The trend is simply what the data is showing if you subtract out all of the other components. 
# 
# The holidays plot shows the effect of all of the holidays included in the model. Holidays, as implemented in Prophet, can be thought of as unnatural events when the trend will deviate from the baseline but return once the event is over.
# 
# The weekly seasonality component shows the change in volume over 
# the week, but with a steep decline on the weekend. 

# In[196]:


#get the predictions
forecast


# In[197]:


#evaluate the predictions
rmse = sqrt(mean_squared_error(test_vol['Volume'], forecast.tail(43)['yhat']))
print(rmse)


# In[198]:


#tabulate the results
resultsDf_temp3 = pd.DataFrame({'Test RMSE': rmse}
                              ,index=['Prophet'])

resultsDf = pd.concat([resultsDf, resultsDf_temp3])
resultsDf


# In[200]:


#plot the graph
plt.plot(train_vol,label='Training Data')
plt.plot(test_vol,label='Test Data')
plt.plot(test_vol.index,pred_SARIMA.predicted_mean,label='Predicted Data - SARIMA')
plt.plot(test_vol.index,pred_SARIMAX.predicted_mean,label='Predicted Data - SARIMAX')
plt.plot(test_vol.index,forecast.tail(43)['yhat'],label='Prophet')
plt.legend(loc='best')
plt.grid()
plt.show()


# If you have independent variables apart from the target forcasted variables, you can add thems as a regressor variables.

# In[201]:


#repeat the same code with additional regressors
prophet=Prophet()
prophet.add_country_holidays(country_name='US')
prophet.add_regressor('High')
prophet.add_regressor('Low')

prophet.fit(df_prophet[df_prophet['ds'] < pd.to_datetime('2019-06-14')])
future = prophet.make_future_dataframe(periods=43, freq=us_bd)
future['High']= df_prophet['High']
future['Low']= df_prophet['Low']
forecast = prophet.predict(future)

fig = prophet.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), prophet, forecast)
plt.show()
fig2 = prophet.plot_components(forecast)
plt.show()


# In[202]:


#calculate the rmse
rmse = sqrt(mean_squared_error(test_vol['Volume'], forecast.tail(43)['yhat']))
print(rmse)


# In[203]:


resultsDf_temp4 = pd.DataFrame({'Test RMSE': rmse}
                           ,index=['Prophet - with exog variables'])

resultsDf = pd.concat([resultsDf, resultsDf_temp4])
resultsDf


# In[ ]:


#we can see that the rmse has reduced significantly


# In[205]:


plt.figure(figsize=(15,8))
plt.plot(train_vol,label='Training Data')
plt.plot(test_vol,label='Test Data')
plt.plot(test_vol.index,pred_SARIMA.predicted_mean,label='Predicted Data - SARIMA')
plt.plot(test_vol.index,pred_SARIMAX.predicted_mean,label='Predicted Data - SARIMAX')
plt.plot(test_vol.index,forecast.tail(43)['yhat'],label='Prophet')
plt.plot(test_vol.index,forecast.tail(43)['yhat'],label='Prophet - with exog variables')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[206]:


#plot test data only with prophet - exog variables
plt.figure(figsize=(15,8))
plt.plot(train_vol,label='Training Data')
plt.plot(test_vol,label='Test Data')
plt.plot(test_vol.index,forecast.tail(43)['yhat'],label='Prophet - with exog variables')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[229]:


#perform cross validation
#Cross Validation of developed time series model. This is a function for cross validation. It creates train and test dataset. Initial days define train data
#and horizon data define test data. Period defines the forecast
from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(prophet, initial = '208 days', period='15 days', horizon='30 days')
df_cv.head()


# In[230]:


#evaluate using performance metrics
from fbprophet.diagnostics import performance_metrics

df_perf = performance_metrics(df_cv)
df_perf.head()


# In[231]:


#plot the metrics
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='rmse')


# Time series model performance: Cross validation performance metrics can be visualized with plot_cross_validation_metric,
# here shown for RMSE. 
# 
# Dots show the error for each prediction in df_cv. The blue line shows the RMSE,
# where the mean is taken over a rolling window of the dots.

# In[ ]:




