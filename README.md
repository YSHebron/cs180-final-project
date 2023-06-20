# cs180-final-project
## Air Quality and COVID-19 Infection Rates

Please see the `cs180-fp.ipynb` file for the final output and `cs180-future-1.ipynb` for further explorations of the problem.

Testing of different models (with hypertuning via GridSearchCV) and preprocessing steps and also the addition of more data have increased our R2 score from 78.3% to 85.3%, but the RMSE score remains poor, reaching as high as 4000, hence future researchers may focus on finding ways to improve on this metric. 

Two web apps were created for this project, a GitHub pages summary of the results and also a Django Webapp that will take in air pollutant concentrations and COVID-19 cases and output the predicted number of new COVID-19 cases.

To run the Django webapp, do `django djangoapp/manage.py runserver` from the root.
