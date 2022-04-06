# %%
import pandas as pd

# %%
events = pd.read_csv('./events.csv', encoding='utf-8')

# %%
events.head()

# %%
events.Date = pd.to_datetime(events.Date, format='%d/%m/%Y')

# %%
events.head()

# %%
events.Date.max()

# %%
events.info()

# %%
events.set_index('Date', inplace=True)

# %%
events.info()

# %% [markdown]
# # Seasonality

# %%
events_filter = events.resample('W').count()

# %%
events_filter.head()

# %%
events_filter.Customer.plot()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
decomp = seasonal_decompose(events_filter.Customer)

# %%
decomp.plot();

# %%
decomp.resid.plot();

# %% [markdown]
# # Autocorrelation

# %%
events_filter.shift()

# %%
events_filter

# %%
import matplotlib.pyplot as plt
plt.style.use('seaborn')

fig, ax = plt.subplots()

events_filter.Customer.plot(ax=ax)
events_filter.Customer.shift(52).plot(ax=ax);

# %%
from scipy.stats import pearsonr

# %%
pearsonr(events_filter.Customer.iloc[52:], events_filter.shift(52).Customer.iloc[52:])

# %%
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16,12))

shift = 4

for row in range(3):
    for col in range(4):
        events_filter.Customer.plot(ax=ax[row][col])
        shifted = events_filter.shift(shift)
        shifted.plot(ax=ax[row][col])
        corr, _ = pearsonr(events_filter.Customer.iloc[shift:], shifted.Customer.iloc[shift:])
        corr = round(corr, 2)
        ax[row][col].set_title(corr)
        shift += 4
fig.tight_layout()

# %%
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# %%
plot_acf(events_filter.Customer, lags=52);

# %% [markdown]
# # Linear Regression

# %%
from scipy import stats
from sklearn.linear_model import LinearRegression

# %%
model = LinearRegression()

# %%
features = events_filter[['Holiday']]
target = events_filter['Customer']

# %%
features

# %%
target

# %%
model.fit(features, target)

# %%
model.intercept_, model.coef_

# %% [markdown]
# So the regression line sklearn has fit to our data has an intercept of 2.17 and a slope of -1.005. So its equation is
# $$\mathrm{Customer Event} = -1.005\times\mathrm{Holiday} - 2.17$$

# %%
fig, axes = plt.subplots()

events_filter.plot(kind='scatter', x='Holiday', y='Customer', alpha=0.2, ax=axes)
axes.plot(events_filter['Holiday'], model.predict(events_filter[['Holiday']]), color='red');

# %%
model.predict([[1]])

# %%
model.score(features, target)

# %% [markdown]
# An r-squared score of 1 means the model is perfect (every prediction it's been tested on is exactly correct). An r-squared score of 0 means the model is performing exactly as well as the baseline model (the mean). A negative r-squared score means your model is worse than the baseline! So r-squared close to 1 is good, close to 0 is bad, negative is terrible. Values above 1 are not possible.
# 
# Mean squared error looks at the error of each prediction (the difference between the predicted value and the true value), squares it (to make it positive and so larger errors are penalised more harshly) and then averages this over all points. MSE is non-negative and the closer to 0 the better!
# 
# Root mean squared error (RMSE) is just the square root of the MSE. It carries the same information, but on the same scale as the data. It is essentially (thought not quite literally) the average amount your predictions are wrong by.
# 
# So the r-squared score is 3% - so our points are 3% less spread out around the red line than they are around the cyan line. The model explains 3% of the variation in the dataset - the remaining 97% is down to other factors (e.g., time of day) and random variation.

# %%
from sklearn.metrics import mean_squared_error as mse

# %%
mse(model.predict(features), target)

# %%
# rmse
mse(model.predict(features), target) ** 0.5

# %% [markdown]
# So our model's predictions are typically wrong by about 2 events on average.
# 
# RMSE has a more direct interpretation in terms of the size of errors (it is roughly the average error), whereas r-squared score is more comparable across different contexts. It is impossible to say what value is good for RMSE without knowing about the context, whereas for r-squared score, good is close to 1, bad is close to 0, and it's much less context-sensitive.

# %%
pearsonr(events_filter.Customer, events_filter.Holiday)

# %% [markdown]
# # ARIMA Forecasting

# %%
events_filter.Customer.plot();

# %%
events_filter.info()

# %%
plot_acf(events_filter.Customer, lags = 45)

# %%
events_filter.Customer.diff().plot();

# %%
plot_acf(events_filter.Customer.diff().dropna());
plot_pacf(events_filter.Customer.diff().dropna());

# %% [markdown]
# in the ACF plot only 3 outside the blue confidence interval in the autocorrelation which is fine, we can legitimately ignore 1 in 20 points outside the 95% confirdence interval. The first is at week 1 and then one each at 8 and 9 weeks but they are only just outside but i have no reason to expect an 8 or 9 week seasonality in my series. This differenced model is now stationary and so an ARMA model can be applied. Times the data was differenced is 1 (so here $d=1$) which will be used to fit the ARIMA model.
# 
# in the PACF plot there are 2 values outside of the confidence interval before the lags are inside the confiedence interval so in this case $p = 2$
# 
# in the ACF plot there is only 1 plot before the lags go inside the confidence interval so in this case $q = 1$
# 
# so our parameters are $p=2, d=1, q=1$ making this an ARIMA (2, 1, 1) model

# %%
events_filter.Customer.shape

# %%
train = events_filter.iloc[:139]
test = events_filter.iloc[139:]

# %%
fig, ax = plt.subplots()

train.Customer.plot(ax=ax, label='train')
test.Customer.plot(ax=ax, label='test')
ax.legend(loc='best');

# %% [markdown]
# # SARIMA Model

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.style.use('seaborn')

# %%
train.Customer

# %%
# model based on train dataset
model = SARIMAX(train.Customer, order=(2, 1, 1)).fit()

# %%
trended_model = SARIMAX(train.Customer, order=(2, 1, 1), trend='t').fit(method='powell', maxiter=500)

# %%
forecast = model.get_forecast(test.shape[0]).summary_frame()
trended_forecast = trended_model.get_forecast(test.shape[0]).summary_frame()

# %%
forecast

# %%
fig, ax = plt.subplots(figsize=(15,8))

# plot the original data
train.Customer.plot(ax=ax, label='train')
test.Customer.plot(ax=ax, label='test')

# plot the forecasts
forecast['mean'].plot(ax=ax, label='forecast', color='red')
trended_forecast['mean'].plot(ax=ax, label='trended_forecast', color='purple')

# shade in the confidence intervals
ax.fill_between(forecast.index, forecast['mean_ci_lower'], forecast['mean_ci_upper'], color='red', alpha=0.1)
ax.fill_between(trended_forecast.index, trended_forecast['mean_ci_lower'], trended_forecast['mean_ci_upper'], color='purple', alpha=0.1)

ax.legend(loc='best');

# %%
from sklearn.metrics import mean_squared_error

print(mean_squared_error(forecast['mean'], test.Customer))
print(mean_squared_error(trended_forecast['mean'], test.Customer))

# %% [markdown]
# suggests normal forecast is better model than the trended forecast because it has a lower mean squared error

# %%
# model based on the entire dataset
final_model = SARIMAX(events_filter.Customer, order=(2,1,1)).fit(method='powell')

# %%
final_forecast = final_model.get_forecast(12).summary_frame()

# %%
fig, ax = plt.subplots(figsize=(15,8))

# plot the original data
events_filter.Customer.plot(ax=ax, label='historic')

# plot the forecasts
final_forecast['mean'].plot(ax=ax, label='forecast', color='red')

# shade in the confidence intervals
ax.fill_between(final_forecast.index, final_forecast['mean_ci_lower'], final_forecast['mean_ci_upper'], color='red', alpha=0.1)

ax.legend(loc='best');

# %% [markdown]
# This forecast is typical to what was expected, it has a wide confidence interval which means the model is not confident in its predictions. Based on the previous data, we can see many points with sharp, big changes in the values. This makes the dataset dificult to forecast.

# %% [markdown]
# # SARIMAX Model

# %%
exog_train = events_filter['Holiday'].iloc[:139]
exog_test = events_filter['Holiday'].iloc[139:]

# %%
holiday_model = SARIMAX(train.Customer, order=(2,1,1), exog=exog_train).fit(method='powell')

# %%
events_filter.index[-1]

# %%
forecast_period = pd.date_range(start="2022-03-06", periods=12, freq="W-SUN")

# %%
exog_forecast = forecast_period.map(events_filter.Holiday).values

# %%
exog_forecast

# %%
holiday_forecast = holiday_model.get_forecast(12, exog=exog_test).summary_frame(alpha=0.1)

# %%
fig, ax = plt.subplots(figsize=(15,8))

# plot the original data
events_filter.Customer.plot(ax=ax, label='historic')

# plot the forecasts
holiday_forecast['mean'].plot(ax=ax, label='forecast', color='red')

# shade in the confidence intervals
ax.fill_between(holiday_forecast.index, holiday_forecast['mean_ci_lower'], holiday_forecast['mean_ci_upper'], color='red', alpha=0.1)

ax.legend(loc='best');

# %%
print(mean_squared_error(holiday_forecast['mean'], test.Customer))

# %%



