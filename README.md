# cs180-final-project
## Air Quality and COVID-19 Infection Rates

Please see the `cs180-fp.ipynb` file for the final output and `cs180-future-1.ipynb` for further explorations of the problem. In particular, the dataset and learning model was improved in `cs180-future-1.ipynb`.

Testing of different models (with hypertuning via GridSearchCV) and additional preprocessing steps have increased our R2 score from 78.3% to 85.3%, but the RMSE score remains poor, reaching as high as 4000. A good R2 score and a poor RMSE score indicates overfitting, hence future researchers may focus on finding ways to improve on this metric.

Two web apps were created for this project: a GitHub summary of the results and a Django webapp that accepts air pollutant concentrations and total COVID-19 cases (treated from the day before) as inputs and outputs the predicted number of new COVID-19 cases for a given day.

To run the Django webapp, do `django djangoapp/manage.py runserver` from the root. Arguments can be passed to the backend through an HTTP POST form or JSON data. Example format:
```
{
  total_cases = 384692,
  confirmed_cases = 384692,
  co = 0.26625,
  no2 = 9.08760,
  o3 = 0.04759,
  pb = 0.01000,
  pm25 = 6.4767,
  so2 = 0.89313
}
```
This particular example should currently return new_cases = 2551. Notice that haven't included the pollutant pm10 in the arguments because we have pruned it from our datasets for not having a significant relationship with new_cases.

Work is still being done on the frontend.
