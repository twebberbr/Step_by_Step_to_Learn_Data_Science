# Project: Analyzing Women's Clothing E-Commerce

# Description

This dataset is commonly used for sentiment analysis, text mining, and customer feedback analysis. It contains reviews written by customers for women's clothing items purchased online. Key elements of the dataset usually include:

- Clothing ID: Unique identifier of the reviewed product.
- Age: Age of the reviewer.
- Title: Title of the review.
- Review Text: The full text of the review.
- Rating: Rating given by the customer, often on a scale of 1 to 5.
- Recommended IND: Indicator if the product is recommended by the reviewer (1 for recommended, 0 for not recommended).
- Positive Feedback Count: Number of other customers who found the review positive.
- Division Name, Department Name, Class Name: Categorical variables indicating the division, department, and class of the clothing item.

## 1. Project Setup

- [✔️] Install necessary Python libraries (pandas, matplotlib, seaborn, etc.).
- [✔️] Load the Women's Clothing E-Commerce Reviews dataset.

## 2. Data Loading and Inspection

- [✔️] Load the dataset using pandas.
- [✔️] Perform basic data inspection (shape, size, data types, info, describe ...)
- [✔️] View the first few rows to understand the data structure.

## 3. Data Cleaning and Preprocessing

- [✔️] Drop irrelevant columns that won’t contribute to the analysis (e.g., 'Unnamed: 0', 'Clothing ID').
- [✔️] Handle missing values by filling or dropping them.
- [✔️] Create derived variables if needed (e.g., word count from review text).
- [✔️] Remove duplicate entries.
