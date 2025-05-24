model_name_map = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "hatebert": "GroNLP/hateBERT",
    "hatexplain":
    "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",
}


def get_car_miles(project_carbon_equivalent: float):
    # code copied from `https://github.com/mlco2/codecarbon`
    """
    8.89 × 10-3 metric tons CO2/gallon gasoline ×
    1/22.0 miles per gallon car/truck average ×
    1 CO2, CH4, and N2O/0.988 CO2
    = 4.09 x 10-4 metric tons CO2E/mile
    = 0.409 kg CO2E/mile
    Source: EPA
    :param project_carbon_equivalent: total project emissions in kg CO2E
    :return: number of miles driven by avg car
    """
    return "{:.4f}".format(project_carbon_equivalent / 0.409)


def get_household_fraction(project_carbon_equivalent: float):
    # code copied from `https://github.com/mlco2/codecarbon`
    """
    Total CO2 emissions for energy use per home: 5.734 metric tons CO2 for electricity
    + 2.06 metric tons CO2 for natural gas + 0.26 metric tons CO2 for liquid petroleum gas
     + 0.30 metric tons CO2 for fuel oil  = 8.35 metric tons CO2 per home per year / 52 weeks
     = 160.58 kg CO2/week on average
    Source: EPA
    :param project_carbon_equivalent: total project emissions in kg CO2E
    :return: % of weekly emissions re: an average American household
    """
    return "{:.4f}".format((project_carbon_equivalent / 160.58) * 100)
