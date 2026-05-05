# oad_zd_heart_disease

## Final Story of the Notebook
This notebook replicates and evaluates whether adding Twitter language topics (2,000 topics) at the county level improves the prediction of age-adjusted atherosclerotic heart disease mortality compared to using traditional demographic and health covariates alone, and also examines if compressing the topics using PCA yields similar or better results.

## Introduction
The project predicts heart disease mortality across US counties by comparing two models: a baseline model and a combined model.
The baseline model ("Traditional only") relies heavily on traditional standard factors:
- income
- education (high school/bachelor graduation variations)
- smoking rates
- diabetes and obesity rates
- hypertension averages
- sex and race demographics (female, black, and hispanic populations)
- marriage rates

The combined model examines whether extending those features with 2,000 text-mined Twitter topics adds predictive accuracy. It tests if the language features encode psychological or community health traits that the structural variables miss.

## Data Acquisition
Data was acquired from county-level outcomes and Twitter topic frequency distributions.

- Outcomes source file: drafts/datasets/countyoutcomes.csv
- Language source files: 
  - drafts/datasets/twitter_topics_county_data1.csv
  - drafts/datasets/twitter_topics_county_data2.csv
- Input structure: counties identified by FIPS (`group_id`), a target variable `ucd_I25_1_atheroHD$0910_ageadj`, 10 traditional variables, and long-format text topic frequencies.

## Data Cleaning
Cleaning and preparation used several steps:

1. Joining Language
- Concatenated the two split Twitter datasets into one.

2. Feature Pivoting
- Pivoted the language dataframe from long format (where 'feat' is the topic and 'value' is its frequency) into a wide format, yielding 2,000 distinct topic features for each county.

3. Missing Data Imputation
- Mean-imputed missing values across the 2,000 language topics.
- Mean-imputed missing values across the 10 traditional covariates.

4. Filtering and Merging
- Dropped counties that lacked a recorded mortality target.
- Standardized the FIPS codes to 5-digit zero-padded integers and inner-joined the language wide-matrix with the outcomes data.

## Model Training

### Models and packages
- **Ridge Regression**: Used `sklearn.linear_model.Ridge`. Optimized via 5-fold cross-validation (`KFold`) minimizing Mean Absolute Error over a logarithmic grid of alphas.
- **Gradient Boosted Trees (GBT)**: Used `sklearn.ensemble.GradientBoostingRegressor` with fixed hyper-parameters (`n_estimators=250`, `learning_rate=0.05`, `max_depth=2`, `subsample=0.8`).
- **Standardization**: Predictors were standard scaled (z-scored). Mean and standard deviation were fit on the training portion and applied equivalently to the test holdout.
- **Dimensionality Reduction**: `sklearn.decomposition.PCA` was fit on the training top-level features.

### Modeling Splits and Feature Sets
The data was split 80/20 train/test.

Models were generated across three different variations of the predictors:
1. Traditional only (10 baseline predictors)
2. All topics (10 baseline + 2,000 topic predictors)
3. PCA topics (10 baseline + compressed topics to maximize efficiency)

## Hypothesis Testing

### Model Comparisons
To mathematically assess improvements, we utilized dependent t-tests on the pair-wise absolute errors for predictions stemming from the Ridge runs.

1. **Did adding topics improve accuracy?**
   - Paired t-test comparing absolute error on the test-set for Ridge (Traditional only) vs Ridge (All topics).
2. **How does standard topics compare to PCA extraction?**
   - Paired t-test comparing absolute error on the test-set for Ridge (All topics) vs Ridge (PCA topics).

## Results

### Metric Comparisons
Models were compared utilizing three metrics. 
1. $R^2$ Score
2. Pearson r correlation
3. Mean Absolute Error (MAE)

Each table row generated displays the performance against the 20% holdout per algorithm variation. Model graphs displaying differences for $R^2$ and MAE visually rank predictive power.

### Meaning of results
Adding the fully expressed 2,000 topics drastically adjusts the Ridge regression accuracy from the baseline traditional model, altering both $R^2$ and the general error layout. The statistical test indicates if the change in error profile achieved significance. When PCA restricts the 2,000 topic elements into compressed dense variants, the modeling footprint shrinks, but affects accuracy and generalization.

## Findings and Conclusion
The prediction outputs confirm whether psychological and linguistic markers observed organically online act as meaningful supplementary variables to standard physical and socioeconomic health risks. 
