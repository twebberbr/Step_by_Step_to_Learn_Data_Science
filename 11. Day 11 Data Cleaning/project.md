# hello there:

## our project is about cleaning the NYC Property Sales Dataset so let's start:

## You can find this dataset on Kaggle [here](https://www.kaggle.com/datasets/new-york-city/nyc-property-sales/data) This data is very dirty, let' make it clean:

## Step 1: Load and Inspect the Data
**Action:** Load the dataset using pandas and inspect the first few rows.  
**Hint:** Use `pd.read_csv()` to load the data. Then, `data.head()` helps you see the first few entries.

## Step 2: Remove Unnecessary Columns
**Action:** Drop columns that are not useful or have constant values.  
**Hint:** Drop columns like `EASE-MENT` which are empty or have a single unique value and `Unnamed: 0`.

## Step 3: whitespace and '-' handling
**Action:** Convert the '-' characters and the whitespace into NaN value to handle them.
**Hint:** you can use the `df.replace()` with the help of a dictionary `{'-':np.nan,' ':np.nan}`

## Step 4: Remove the duplicated values
**Action:** Remove the duplicated values
**Hint:** Use `df.drop_duplicates` and `df.duplicated`

## Step 5: Identify Missing Values
**Action:** Check for missing values in the dataset.  
**Hint:** Use `df.isnull().sum()` to identify columns with missing values.

## Step 6: Handle Non-Numeric Data in Numeric Columns
**Action:** Convert columns with numeric values stored as object  `numeric_columns = ['LAND SQUARE FEET', 'GROSS SQUARE FEET', 'SALE PRICE']` into numeric types (e.g., float or int).  
**Hint:** Use `df[col].astype('float')` to convert and handle non-numeric data.

## Step 7: Impute Missing Values
**Action:** Fill missing values in important columns.  
**Hint:** Use `df[col].fillna` and  ` df.dropna`
- For real estate data, use the median to impute missing values in `LAND SQUARE FEET` and `GROSS SQUARE FEET`.
- For columns like `SALE PRICE`, consider dropping rows with missing values instead of imputing.

## Step 8: Check the data consistency
**Action** convert the data type into datetime then divied it into 3 columns the first for years and the seconde for the months, the last for the hours
**Hint** Use `pd.to_datetime(df['SALE DATE'])` and `df['year'] = df['SALE DATE'].dt.year`

## Step 9: Save the Cleaned Data
**Action:** Save the cleaned dataset for further analysis.  
**Hint:** Use `df.to_csv()` to save your cleaned dataset to a new CSV file.
