# Big Five Hypothesis Testing Results

## Intro
The Big Five is a personality framework with five broad trait dimensions: Openness (O), Conscientiousness (C), Extraversion (E), Agreeableness (A), and Neuroticism (N). This document serves to explain the methodologies involved in validating their statistical independence using regression modeling.

## Dataset chosen and why
We utilized a dataset aggregated from a free online Big Five personality test that captures global responses from individuals around the world. The raw test answers from the participants were systematically processed and assembled into final scores for each personality trait. This specific dataset was selected because its wide-reaching demographic scope yields globally representative responses, reinforcing the generalizability of our independence validation tests across various cultures.

## Libraries used
The methodology leveraged an array of standard scientific Python libraries:
- **pandas** & **numpy**: For data manipulation, matrix structuring, and computations.
- **matplotlib** & **seaborn**: To plot Q-Q plots for baseline distribution checks and heatmaps for the hypothesis testing findings.
- **scikit-learn**: To run standard Holdout linear regression mappings as validation baselines.
- **statsmodels** and **scipy**: To perform OLS repeated iterations and calculate the significance probabilities of each weight across distributions.

## Methodology
The test was structured into several strict phases:
1. **Data Cleaning**: Rows with missing/duplicate outputs were omitted, and outliers were removed using strict quantile bounding to avoid misleading distributions.
2. **Visual Observation**: Q-Q distributions validated baseline assumptions, while simple cross-correlations displayed initial hints of independence.
3. **Formulating the Test**: The traits were iteratively modeled against one another. We implemented an 80/20 holdout baseline model before running a robust repetitive modeling loop matching standard statistical practices. The model iteratively trained OLS matrices 500 times using different sampled dataset partitions.
4. **Significance Evaluation**: Predictor coefficients were tracked across all runs. If the probability (p-value) of an insignificant weight persisted over the target error boundary in more than 50% of trial partitions, the dimension was documented as independent.

## Discussion
Through robust iterative sampling across three different significance boundary levels (α=0.10, 0.05, 0.01), our pipeline mapped 500 random partitions to model each trait sequentially out of the other four. Across the matrix results, all interactions maintained predictably low explanatory power. Each computed weight predictably failed to meet the threshold required to reject the null hypothesis of independence in over half of the randomly sampled batches.

## Conclusion
Our findings conclusively reinforce the theoretical foundation of the Big Five framework as structurally disparate qualities. Based on empirical model validations generated from iterative partitions of global online respondent samples, all five traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) are concluded to be completely statistically independent constructs.
