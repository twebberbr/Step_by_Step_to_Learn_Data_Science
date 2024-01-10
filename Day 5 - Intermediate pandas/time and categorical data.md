## Time Series Data Handling Functions

1. **`pd.to_datetime`**

   - Converts strings or numbers to DateTime.
   - `pd.to_datetime(df['date_column'])`

2. **`DataFrame.resample`**

   - Aggregates data over a time interval.
   - `df.resample('D').mean()` (daily mean)

3. **`DataFrame.asfreq`**

   - Changes the frequency of time series.
   - `df.asfreq('M')` (monthly frequency)

4. **`DataFrame.shift`**

   - Shifts data by periods.
   - `df.shift(1)` (shift by one period)

5. **`DataFrame.tz_localize` & `DataFrame.tz_convert`**

   - Time zone localization and conversion.
   - `df.tz_localize('UTC').tz_convert('US/Eastern')`

6. **`pd.date_range`**
   - Creates a DatetimeIndex with fixed frequency.
   - `pd.date_range('2020-01-01', periods=10, freq='D')`

## Handling Categorical Data Functions

1. **`astype('category')`**

   - Converts a column to categorical type.
   - `df['column_name'].astype('category')`

2. **`pd.get_dummies`**

   - Converts categories to dummy variables.
   - `pd.get_dummies(df['category_column'])`

3. **`Series.cat.categories`**

   - Lists categories in a series.
   - `df['category_column'].cat.categories`

4. **`Series.cat.codes`**

   - Gets integer codes for categories.
   - `df['category_column'].cat.codes`

5. **`DataFrame.groupby`**

   - Groups data by categories.
   - `df.groupby('category_column').mean()`

6. **`Series.value_counts`**
   - Counts unique values in a series.
   - `df['category_column'].value_counts()`

---

These functions are essential for handling time series and categorical data in pandas, enabling effective data analysis and manipulation.

---
