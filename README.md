# Employee Attrition Prediction Pipeline using Naive Bayes

This project uses a pipeline with a Naive Bayes classifier to predict employee attrition. The model helps identify employees likely to leave the company based on key factors such as job satisfaction, number of projects, working hours, and tenure. Using a Scikit-Learn pipeline enables efficient data pre-processing and model training.

## Project Overview

Employee attrition can negatively impact an organization due to increased costs for hiring and training new staff. Predicting attrition enables companies to implement strategies to improve employee satisfaction and reduce turnover rates. This project uses a Naive Bayes model with a pipeline structure for streamlined data processing and prediction.

The pipeline includes pre-processing steps for scaling numeric data, which enhances the model's ability to handle features of varying ranges.

## Dataset

This project is based on a synthetic dataset with the following features:

- **Job_Satisfaction**: Satisfaction level of the employee (scale of 1 to 5)
- **Num_Projects**: Number of projects the employee is handling
- **Working_Hours_Per_Week**: Average weekly working hours
- **Years_at_Company**: Total years the employee has been with the company
- **Attrition**: Target variable, where 1 indicates "likely to leave" and 0 indicates "likely to stay"

## Pipeline Overview

1. **Data Preprocessing**: The pipeline uses `StandardScaler` to normalize numerical features, which is important for improving model accuracy.
2. **Classification Model**: Naive Bayes (`GaussianNB`) is used to predict employee attrition based on pre-processed features.

## Installation

To run this project, clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
