import numpy as np
import pandas as pd
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response


class Prediction(APIView):
    def post(self, request):
        #data = request.data
        total_cases = request.POST.get('total_cases')
        confirmed_cases = request.POST.get('confirmed_cases')
        co = request.POST.get('co')
        no2 = request.POST.get('no2')
        o3 = request.POST.get('o3')
        pb = request.POST.get('pb')
        # pm10 = request.POST.get('pm10')
        pm25 = request.POST.get('pm25')
        so2 = request.POST.get('so2')
        regressor = ApiConfig.model
        #predict using independent variables
        print(total_cases, confirmed_cases, co, no2, o3, pb, pm25, so2)
        PredictionMade = regressor.predict([[total_cases, confirmed_cases, co,
                                          no2, o3, pb, pm25, so2]])
        response_dict = {"New Cases": PredictionMade}
        print(response_dict)
        return Response(response_dict, status=200)
