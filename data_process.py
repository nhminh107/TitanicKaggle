import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
class DataProcess:
    def __init__(self, data, data_test, avgAge=-1, finalCabinDict=None):
        self.data = data
        self.data_test = data_test
        self.avgAge = avgAge
        self.finalCabinDict = finalCabinDict if finalCabinDict is not None else {}
        self.imputer = SimpleImputer(strategy='mean')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def dropColumn(self): 
        self.data.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
        self.data_test.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
    def fillCabinTrain(self): 
        most_freq_cabins = self.data.dropna(subset=['Cabin']).groupby('Pclass')['Cabin'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
        self.finalCabinDict = most_freq_cabins
        
        fill_values = self.data['Pclass'].map(self.finalCabinDict)
        self.data['Cabin'] = self.data['Cabin'].fillna(fill_values)

    def convertCabinName(self):
        self.data['Cabin'] = self.data['Cabin'].str[:1]
        self.data_test['Cabin'] = self.data_test['Cabin'].str[:1]
    def fillAge(self):
        if self.avgAge == -1: 
            self.avgAge = self.data['Age'].mean()
        self.data['Age'] = self.data['Age'].fillna(self.avgAge)

        self.data_test['Age'] = self.data_test['Age'].fillna(self.avgAge)
    
    def fillCabinTest(self): 
        fill_values = self.data_test['Pclass'].map(self.finalCabinDict)
        self.data_test['Cabin'] = self.data_test['Cabin'].fillna(fill_values)

    def splitData(self): 
        X_train = self.data.drop(['Survived'], axis = 1)
        X_test = self.data_test
        y_train = self.data['Survived']

        return (X_train, y_train, X_test)

    def feature_engineering(self):
        for ds in [self.data, self.data_test]:
            ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            ds['Title'] = ds['Title'].replace(
                ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            ds['Title'] = ds['Title'].replace(['Mlle', 'Ms'], 'Miss')
            ds['Title'] = ds['Title'].replace('Mme', 'Mrs')

            ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
            ds['IsAlone'] = 0
            ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
            ds.drop(['Name', 'SibSp', 'Parch'], axis=1, inplace=True, errors='ignore')
    def encoder_categorical(self):
        cols = ['Title', 'Sex', 'Embarked', 'Cabin']

        train_encoded = self.encoder.fit_transform(self.data[cols])
        test_encoded = self.encoder.transform(self.data_test[cols])

        train_encoded_df = pd.DataFrame(train_encoded, columns=self.encoder.get_feature_names_out(cols))
        test_encoded_df = pd.DataFrame(test_encoded, columns=self.encoder.get_feature_names_out(cols))

        self.data = pd.concat([self.data.drop(cols, axis=1), train_encoded_df], axis=1)
        self.data_test = pd.concat([self.data_test.drop(cols, axis=1), test_encoded_df], axis=1)

    def convertCSV(self, passenger_ids,predictions):
        CSV_File = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        return CSV_File
     

def main():
    data = pd.read_csv('Titanic/data/train.csv')
    processor = DataProcess(data)
    processor.fillCabinTrain()

if __name__ == '__main__':
    main()