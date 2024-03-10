# Investigating Uncertainty in Evaluation Metrics of Ensemble Flood Modeling
These Python scripts are developed for investigating and demonstrating the sampling uncertainty in several popular evaluation metrics (uncertainty bounds, NSE, KGE, and R<sup>2</sup>, which are concerted to the uncertainty coefficients, UCs) of ensemble flood modeling by using bootstrapping Analysis. For more information, please refer to the paper, “[<b><i>Beyond a fixed number: Investigating uncertainty in popular evaluation metrics of ensemble flood modeling using bootstrapping analysis</b></i>]((https://onlinelibrary.wiley.com/doi/full/10.1111/jfr3.12982))” (Huang and Merwade, 2024).

A brief introduction to the features of each Python script is as follows.

(1) [1-Ensemble_Flood_Modeling.py](https://github.com/huan1441/Investigating-Uncertainty-in-Evaluation-Metrics-of-Ensemble-Flood-Modeling/blob/main/1-Ensemble_Flood_Modeling.py) is developed to generate the ensemble of HEC-RAS configurations accounting for Manning’s n & upstream flow (Q) and output the top model members with the smallest sum of square errors.

(2) [2-UCs_under_Different_High_Scenarios.py](https://github.com/huan1441/Investigating-Uncertainty-in-Evaluation-Metrics-of-Ensemble-Flood-Modeling/blob/main/2-UCs_under_Different_High_Scenarios.py) is developed to investigating the impact of different high-flow scenarios on the evaluation metrics (UCs) by using Bootstrapping analysis.

(3) [3-Mean_CI_of_UCs.py](https://github.com/huan1441/Investigating-Uncertainty-in-Evaluation-Metrics-of-Ensemble-Flood-Modeling/blob/main/3-Mean_CI_of_UCs.py) is developed to estimate the means and confidence intervals of UCs obtained based on different priors.

(4) [4-Errors_in_Observed_Data.py](https://github.com/huan1441/Investigating-Uncertainty-in-Evaluation-Metrics-of-Ensemble-Flood-Modeling/blob/main/4-Errors_in_Observed_Data.py) is developed to investigating the impact of the measurement errors of observed hydrologic data on the evaluation metrics (UCs).
