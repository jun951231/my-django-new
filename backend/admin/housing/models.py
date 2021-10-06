import matplotlib.pyplot as plt
import numpy as np
from django.db import models
from icecream import ic
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from admin.common.models import DFrameGenerator
import pandas as pd


class HousingService(object):


    def __init__(self):
        self.dfg = DFrameGenerator()
        self.dfg.fname = 'admin/housing/data/housing.csv'
        self.df = self.dfg.create_model()

    def housing_info(self):
        self.dfg.model_info(self.df)

    def housing_hist(self):
        self.model.dframe.hist(bins=50, figsize=(20, 15))
        plt.savefig('admin/housing/data/housing.csv')

    def split_model(self) -> []:
        train_set, test_set = train_test_split(self.model, test_size=0.2, random_state=42)
        print('#'*100)
        self.dfg.model_info(train_set)
        print('#'*100)
        self.dfg.model_info(test_set)
        return [train_set, test_set]

    def income_cat_hist(self):
        m = self.model
        m['income_cat'] = pd.cut(m['median_income'],
                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1,2,3,4,5]
                                 )
        m['income_cat'].hist()
        plt.savefig('admin/housing/image/housing-hist.png')

    def split_model_by_income_cat(self) -> []:
        m = self.model
        split = StratifiedShuffleSplit(n_splits=1, test=0.2, random_state=42)
        for train_idx, test_idx in split.split(m, m['income_cat']):
            temp_train_set = m.loc[train_idx]
            temp_test_set = m.loc[test_idx]
        ic(temp_test_set['income_cat']).value_counts() / len(temp_test_set)

class Housing(models.Model):

    housing_id = models.AutoField(primary_key=True)
    latitude = models.FloatField()
    housing_median_age = models.FloatField()
    total_rooms = models.FloatField()
    total_bedrooms = models.FloatField()
    population = models.FloatField()
    households = models.FloatField()
    median_income = models.FloatField()
    median_house_value = models.FloatField()
    ocean_proximity = models.TextField()

    class Meta:
        db_table = "housing"

    def __str__(self):
        return f'[{self.pk}] {self.housing_id}'