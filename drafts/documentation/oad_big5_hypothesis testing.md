# oad_big5_hypothesis testing

## Final Story of the Notebook
This notebook evaluates whether Big Five personality traits are highly dependent on each other or mostly distinct, using cleaned psychometrics test data, linear models, and repeated hypothesis testing.

## Introduction
The Big Five personality framework describes personality with five broad traits:

1. Agreeableness (A): cooperation, empathy, trust.
2. Openness (O): curiosity, imagination, interest in ideas.
3. Conscientiousness (C): organization, discipline, reliability.
4. Extraversion (E): sociability, assertiveness, energy.
5. Neuroticism (N): emotional instability and stress reactivity.

The notebook asks whether each trait can be strongly predicted by the other four. If prediction is weak, traits are related but still mostly distinct.

## Data Acquisition
Data was acquired from online psychometrics Big Five test scores.

- Source file: drafts/datasets/big5-dataset.csv
- Input structure: item-level responses (A1-A10, O1-O10, C1-C10, E1-E10, N1-N10) plus metadata.

## Data Cleaning
Cleaning and preparation used four steps:

1. Base cleaning
- Removed duplicates.
- Removed rows with missing values.

2. Trait construction
- Built trait totals:
  - A_total from A1-A10
  - O_total from O1-O10
  - C_total from C1-C10
  - E_total from E1-E10
  - N_total from N1-N10

3. Strict tail cleaning
- Applied trait-only quantile filtering:
  - lower bound: 1st percentile
  - upper bound: 97.5th percentile
- Kept rows that satisfy all five trait bounds.

4. Standardization
- Applied z-score scaling to cleaned trait totals.

Row impact:
- rows before strict cleaning: 19,710
- rows after strict cleaning: 17,387
- rows removed: 2,323 (11.79%)

## Data Visualization
Two diagnostics were used before and after strict cleaning:

1. Histograms of trait totals.
2. Q-Q plots per trait.

Observed pattern:
- Points were nearly on the reference line through most of the distribution.
- Deviations appeared mainly at tails (upper and lower outliers).
- This justified stricter tail handling.

Post-cleaning correlation matrix result:
- Trait correlations remained mostly small to moderate, approximately from -0.05 to 0.21.

## Model Training
### Models and packages
- sklearn.linear_model.LinearRegression for predictive modeling.
- sklearn.model_selection.train_test_split and sklearn.metrics.r2_score for holdout evaluation.
- statsmodels.api.OLS for inferential regression and predictor-level hypothesis tests.

### Method A: 80/20 holdout validation
- Split standardized trait table into 80% train and 20% test.
- For each target trait, use the other four traits as predictors.
- Report train and test R2.

Holdout test R2 results:
- A_total: 0.072634
- C_total: 0.091449
- E_total: 0.056916
- N_total: 0.036924
- O_total: 0.064379

These values are modest, indicating limited out-of-sample predictability.

### Method B: repeated sampling and model stability
- Repeated OLS workflow with 500 repetitions.
- Each repetition picks random 70 rows (without replacement inside that repetition).
- Sampling audit confirms exactly 500 samples and exactly 70 unique rows per sample.

## Hypothesis Testing
### Define the null hypothesis
Predictor-level test for each target model:

- H0: beta = 0 (predictor has no linear contribution to target, conditional on other predictors).
- H1: beta != 0.

Interpretation in this study:
- If coefficients are near 0 and frequently non-significant, that predictor has weak dependency contribution for that target.

### Choose a test
Two equivalent inferential views were used:

1. coefficient p-values from OLS.
2. partial F-tests from reduced-vs-full model comparisons.

### Set significance levels
- alpha = 0.10
- alpha = 0.05
- alpha = 0.01

### Compute critical values
Implementation uses p-values directly.
Equivalent critical-value decision rules are:

- reject H0 if |t| > t_critical(alpha/2, df), or
- reject H0 if F > F_critical(alpha, df1=1, df2).

### Decision logic at each alpha
For each target-predictor pair across 500 repetitions:

1. Calculate non-significance rate at each alpha.
2. Summarize coefficient mean and standard deviation.
3. Summarize partial R2 with confidence interval.
4. Mark practical non-importance using threshold partial R2 < 0.01.

Decision interpretation:
- high non-significance rate + near-zero coefficient -> weak importance,
- low non-significance rate + larger effect size -> stronger importance.

## Results
### Overall repeated-model fit across targets (500 repetitions)
- A_total: mean R2 = 0.130350, mean adjusted R2 = 0.076833
- C_total: mean R2 = 0.130899, mean adjusted R2 = 0.077416
- E_total: mean R2 = 0.104598, mean adjusted R2 = 0.049497
- N_total: mean R2 = 0.095527, mean adjusted R2 = 0.039867
- O_total: mean R2 = 0.107686, mean adjusted R2 = 0.052775

### Meaning of results
1. Predictive power is limited to modest.
2. Most target variance remains unexplained by the other four traits.
3. Traits show measurable overlap, but not strong redundancy.

## Findings and Conclusion
1. Q-Q plots were close to normal in the center, with tail outliers motivating strict cleaning.
2. Strict cleaning improved robustness while keeping most rows.
3. Holdout and repeated-model results agree: dependency exists but is modest.
4. Statistical evidence supports the conclusion that Big Five traits are related yet largely distinct dimensions.
5. The notebook is now fully documented from acquisition to inference, with explicit sampling audit and decision rules.
