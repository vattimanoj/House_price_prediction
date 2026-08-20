import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys
class HOUSE_PRICE_PREDICTION:
    def __init__(self,path):
        try:
            self.df = pd.read_csv(path)
            self.df['date'] = pd.to_datetime(self.df['date'])

        # Keep only day, month, year
            self.df['sale_day'] = self.df['date'].dt.day
            self.df['sale_month'] = self.df['date'].dt.month
            self.df['sale_year'] = self.df['date'].dt.year

            # Drop the now-redundant raw date column
            self.df = self.df.drop('date',axis=1)
            self.df['city'] = self.df['city'].astype('category').cat.codes
            self.df['country'] = self.df['country'].astype('category').cat.codes
            self.x=self.df.iloc[:,1:]
            self.y=self.df.iloc[:,0]

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_ty}")
    def train_data(self ):
        try:
            print("training data")
            self.reg=LinearRegression()
            self.reg.fit(self.X_train,self.y_train)
            a=self.reg.coef_
            b=self.reg.intercept_
            print(f"m_value:{a}")
            print(f"c_value:{b}")
            train_pred=self.reg.predict(self.X_train)
            #r2score
            self.y_mean=self.y_train.mean()
            numerator=((self.y_train-train_pred)**2).sum()
            denominator=((self.y_train-self.y_mean)**2).sum()
            r2=1-(numerator/denominator)
            print(f"r2_score : {r2}")
            #mse
            mse=((self.y_train-train_pred)**2).sum()/len(self.y_train)
            print(f"mse : {mse}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_ty}")
    def test_data(self):
        try:
            print("testing data")
            test_pred=self.reg.predict(self.X_test)
            y_test_mean=self.y_test.mean()
            numerator=((self.y_test-test_pred)**2).sum()
            denominator=((self.y_test-y_test_mean)**2).sum()
            r2=1-(numerator/denominator)
            print(f"r2_score : {r2}")
            mse=((self.y_test-test_pred)**2).sum()/len(self.y_test)
            print(f"mse : {mse}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_ty}")
        with open("MLR.pkl", "wb") as f:
            pickle.dump(self.reg, f)

if __name__ == "__main__":
    odj=HOUSE_PRICE_PREDICTION("house price prediction.csv")
    odj.train_data()
    odj.test_data()



