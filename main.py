from model import Model
from data_process import DataProcess 
import pandas as pd 

def main(): 
    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    
    passenger_ids = data_test['PassengerId'].copy()
    
    Processer = DataProcess(data_train, data_test, -1, None) 
    md = Model(isRunning=False) 

    Processer.dropColumn()
    Processer.fillAge() 
    Processer.fillCabinTrain()
    Processer.fillCabinTest()
    Processer.feature_engineering()
    Processer.encoder_categorical()

    X_train, y_train, X_test = Processer.splitData()
    
    md.fit(X_train, y_train)
    y_pred = md.predict(X_test) 
    
    submission = Processer.convertCSV(passenger_ids, y_pred)
    submission.to_csv('submissionTitanic.csv', index=False)

if __name__ == '__main__':
    main()