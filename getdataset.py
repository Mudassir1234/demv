from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np

#return df, label, positive_label, sensitive_features, unpriv_group


def getdataset(dataset, numberoffeatures, singlefeature = None):
    if(dataset == 'adult'):
        lab_enc = LabelEncoder()
        ord_enc = OrdinalEncoder()
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country', 'income']
        adult_df = pd.read_csv('data/adult.data', names=column_names)
        adult_df.drop(adult_df[(adult_df['race'] != ' Black') & (
            adult_df['race'] != ' White')].index, inplace=True)
        adult_df.loc[adult_df['native-country'] ==
                        ' ?', 'native-country'] = 'Not known'
        adult_df['age_class'] = pd.cut(adult_df['age'],
                                        bins=[0, 9, 19, 29, 39, 49, 59, 69, 99],
                                        labels=['age<10', 'age between 10 and 20', 'age between 20 and 30',
                                                'age between 30 and 40', 'age between 40 and 50',
                                                'age between 50 and 60', 'age between 60 and 70', 'age>70']
                                        )
        adult_df['hour-per-week-class'] = pd.cut(adult_df['hours-per-week'],
                                                    bins=[0, 9, 19, 29, 39, 49, 99],
                                                    labels=['hour<10', 'hours between 10 and 20', 'hours between 20 and 30',
                                                            'hours between 30 and 40', 'hour between 40 and 50',
                                                            'hour>70']
                                                    )
        adult_df.drop(labels=['hours-per-week', 'workclass', 'fnlwgt', 'capital-gain', 'capital-loss', 'age', 'education-num'],
                        axis=1, inplace=True)
        adult_df = adult_df.join(pd.get_dummies(
            adult_df['education'])).drop('education', axis=1)
        adult_df = adult_df.join(pd.get_dummies(
            adult_df['marital-status'])).drop('marital-status', axis=1)
        adult_df = adult_df.join(pd.get_dummies(
            adult_df['occupation'])).drop('occupation', axis=1)
        adult_df = adult_df.join(pd.get_dummies(
            adult_df['relationship'])).drop('relationship', axis=1)
        adult_df = adult_df.join(pd.get_dummies(
            adult_df['native-country'])).drop('native-country', axis=1)
        adult_df = adult_df.join(pd.get_dummies(
            adult_df['age_class'])).drop('age_class', axis=1)
        adult_df = adult_df.join(pd.get_dummies(
            adult_df['hour-per-week-class'])).drop('hour-per-week-class', axis=1)
        adult_df['income'] = lab_enc.fit_transform(adult_df['income'])
        adult_df[['sex', 'race']] = ord_enc.fit_transform(
            adult_df[['sex', 'race']])

        adult_df.rename(columns={" Bachelors": "Bachelors", "hour<10": "hours"}, inplace=True)
        label = 'income'
        positive_label = 1
        k = 200

        if numberoffeatures == 1:
            if singlefeature != 2:
                unpriv_group = {'sex':0}
            else:
                unpriv_group = {'race':0}
        elif numberoffeatures == 2:
            unpriv_group = {'sex':0, 'race':0}
        elif numberoffeatures == 3:
            unpriv_group = {'sex':0, 'race':0, 'Bachelors':0}
        elif numberoffeatures == 4:
            unpriv_group = {'sex':0, 'race':0, 'Bachelors':0, 'hours':0}
        else:
            print(" ERROR: Wrong number of features. ")


        sensitive_features = unpriv_group.keys()
        return adult_df, label, positive_label, sensitive_features, unpriv_group, k

    elif(dataset == 'cmc'):

        data = pd.read_csv('data/cmc.data', names=['wife_age', 'wife_edu', 'hus_edu', 'num_child', 'wife_religion', 'wife_work', 'hus_occ', 'living', 'media', 'contr_use'])
        label = 'contr_use'
        sensitive_features = ['wife_religion', 'wife_work']
        unpriv_group = {'wife_religion': 1, 'wife_work': 1}
        positive_label= 2
        k = 3

        if numberoffeatures == 1:
            if singlefeature != 2:
                unpriv_group = {'wife_religion': 1}
            else:
                unpriv_group = {'wife_work' : 1}
        elif numberoffeatures == 2:
            unpriv_group = {'wife_religion': 1, 'wife_work': 1}
        elif numberoffeatures == 3:
            unpriv_group = {'wife_religion': 1, 'wife_work': 1, 'wife_edu':0}
            key = 'wife_edu'
            threshold = 33
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1
        elif numberoffeatures == 4:
            unpriv_group = {'wife_religion': 1, 'wife_work':1, 'wife_edu':0, 'hus_occ':0}
            threshold = {'wife_edu':33, 'hus_occ':3}
            
            key = 'wife_edu'
            threshold = 33
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'hus_occ'
            threshold = 3
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        sensitive_features = unpriv_group.keys()
        return data, label, positive_label, sensitive_features, unpriv_group, k


    elif(dataset == 'compas'):

        data = pd.read_csv('data/compas.csv', index_col=0)
        label = 'two_year_recid'
        positive_label = 1
        k = 29

        if numberoffeatures == 1:
            if singlefeature != 2:
                protected_group = {'sex':0}
            else:
                protected_group = {'race': 0 }
        elif numberoffeatures == 2:
            protected_group = {'sex':0, 'race':0}
        elif numberoffeatures == 3:
            protected_group = {'sex':0, 'race':0, 'age':0}
            key = 'age'
            threshold = 50
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1
        elif numberoffeatures == 4:
            raise Exception("COMPAS only allows 3 sensitive variables.")

        sensitive_vars = protected_group.keys()

        return data, label, positive_label, sensitive_vars, protected_group, k 

    elif dataset == 'crime':

        data = pd.read_excel('data/crime_data_normalized.xlsx', na_values='?')
        data.drop(['state', 'county', 'community', 'communityname',
                    'fold', 'OtherPerCap'], axis=1, inplace=True)
        na_cols = data.isna().any()[data.isna().any() == True].index
        data.drop(na_cols, axis=1, inplace=True)
        data = (data - data.mean())/data.std()
        y_classes = np.quantile(data['ViolentCrimesPerPop'].values, [
                                0, 0.2, 0.4, 0.6, 0.8, 1])
        i = 0
        data['ViolentCrimesClass'] = data['ViolentCrimesPerPop']
        for cl in y_classes:
            data.loc[data['ViolentCrimesClass'] <= cl, 'ViolentCrimesClass'] = i*100
            i += 1
        data.drop('ViolentCrimesPerPop', axis=1, inplace=True)
        data['black_people'] = data['racepctblack'] > -0.45
        data['hisp_people'] = data['racePctHisp'] > -0.4
        data['black_people'] = data['black_people'].astype(int)
        data['hisp_people'] = data['hisp_people'].astype(int)
        data.drop('racepctblack', axis=1, inplace=True)
        data.drop('racePctHisp', axis=1, inplace=True)

        label = 'ViolentCrimesClass'
        positive_label = 100
        k = 41

        if numberoffeatures == 1:
            if singlefeature != 2:
                groups_condition = {'black_people': 1}
            else:
                groups_condition = {'hisp_people': 1}
        if numberoffeatures == 2:
            groups_condition = {'black_people': 1, 'hisp_people':1}
        if numberoffeatures == 3:
            groups_condition = {'black_people':1, 'hisp_people':1, 'MedRent':1}

            key = 'MedRent'
            threshold = 0.1
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        if numberoffeatures == 4:
            groups_condition = {'black_people':1, 'hisp_people':1, 'MedRent':1, 'racePctAsian':0}

            key = 'MedRent'
            threshold = 0.1
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'racePctAsian'
            threshold = 1.1
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1



        sensitive_features = groups_condition.keys()
        return data, label, positive_label, sensitive_features, groups_condition, k

    elif(dataset == 'drug'):

        data = pd.read_csv('data/drugs.csv')
        data.drop(['yhat','a'], axis=1, inplace=True)
        data.loc[data['gender']==0.48246,'gender']=1
        data.loc[data['gender']==-0.48246,'gender']=0
        data['y'].replace({
            'never': 0,
            'not last year': 1,
            'last year': 2}, inplace=True)
        data['race'].replace({
            'non-white': 0,
            'white': 1}, inplace=True)
        string_cols = data.dtypes[data.dtypes == 'object'].index.values
        data.drop(string_cols, axis=1, inplace=True)

        label = 'y'
        protected_group = {'race': 1, 'gender': 0}
        positive_label = 0
        sensitive_features = ['race', 'gender']
        k = 22

        if numberoffeatures == 1:
            if singlefeature != 2:
                protected_group = {'race':1}
            else:
                protected_group = {'gender': 0 }
        if numberoffeatures == 2:
            protected_group = {'race': 1, 'gender': 0}
        if numberoffeatures == 3:
            protected_group = {'race': 1, 'gender': 0, 'age':1}

            key = 'age'
            threshold = 0
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1
            
        if numberoffeatures == 4:
            protected_group = {'race': 1, 'gender': 0, 'age':1, 'country':0}

            key = 'age'
            threshold = 0
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'country'
            threshold = 0.1
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1
        
        sensitive_features = protected_group.keys()

        return data, label, positive_label, sensitive_features, protected_group, k

    elif(dataset == 'german'):

        data = pd.read_csv('data/german.csv', index_col=0)

        label = 'credit'
        positive_label = 1
        sensitive_features = ['sex', 'age']
        unpriv_group = {'sex': 0, 'age': 0}

        if numberoffeatures == 1:
            if singlefeature != 2:
                unpriv_group = {'sex' : 0}
            else:
                unpriv_group = {'age':0}
        if numberoffeatures == 2:
            unpriv_group = {'sex':0, 'age':0}
        if numberoffeatures == 3:
            unpriv_group = {'sex':0, 'age':0, 'investment_as_income_percentage' :0}

            key = 'investment_as_income_percentage'
            threshold = 3
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        if numberoffeatures == 4:

            unpriv_group = {'sex':0, 'age':0, 'investment_as_income_percentage' :0, 'month':0}

            key = 'investment_as_income_percentage'
            threshold = 3
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'month'
            threshold = 30
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1


        k = 2
        sensitive_features = unpriv_group.keys()
        return data, label, positive_label, sensitive_features, unpriv_group, k


    elif(dataset == 'law'):

        data = pd.read_csv('data/bar_pass_prediction.csv', index_col='Unnamed: 0')
        col_to_drop = ['ID', 'decile1b', 'decile3', 'decile1', 'cluster', 'bar1', 'bar2',
                    'sex', 'male', 'race1', 'race2', 'other', 'asian', 'black', 'hisp', 'bar', 'index6040', 'indxgrp', 'indxgrp2', 'dnn_bar_pass_prediction', 'grad', 'bar1_yr', 'bar2_yr', 'ugpa']
        data.drop(col_to_drop, axis=1, inplace=True)
        data.loc[data['Dropout'] == 'NO', 'Dropout'] = 0
        data.loc[data['Dropout'] == 'YES', 'Dropout'] = 1
        data['Dropout'] = data['Dropout'].astype(np.int32)
        data.dropna(inplace=True)
        data.loc[data['gender']=='female', 'gender'] = 1
        data.loc[data['gender'] == 'male', 'gender'] = 0
        data['gender'] = data['gender'].astype(np.int32)
        data.loc[data['race']==7.0, 'race'] = 0
        data.loc[data['race'] != 0, 'race'] = 1
        data['gpa'] = pd.qcut(data['gpa'], 3, labels=['a','b','c'])
        enc = LabelEncoder()
        data['gpa'] = enc.fit_transform(data['gpa'].values)

        label = 'gpa'
        positive_label = 2

        if numberoffeatures == 1:
            if singlefeature != 2:
                protected_group = {'race':1}
            else:
                protected_group = {'gender': 1}
        if numberoffeatures == 2:
            protected_group = {'race':1, 'gender':1}
        if numberoffeatures == 3:
            protected_group = {'race':1, 'gender':1, 'age':0}

            key = 'age'
            threshold = 61
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        if numberoffeatures == 4:
            protected_group = {'race':1, 'gender':1, 'age':0, 'fam_inc':0}

            key = 'age'
            threshold = 61
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'fam_inc'
            threshold = 3
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1



        k = 103
        sensitive_features = protected_group.keys()
        return data, label, positive_label, sensitive_features, protected_group, k 

    elif(dataset == 'obesity'):

        data = pd.read_csv('data/obesity.csv')
        data.drop(['NObeyesdad', 'weight_cat', 'yhat', 'a'], axis=1, inplace=True)
        le = LabelEncoder()
        data['Gender'] = le.fit_transform(data['Gender'].values)
        data['y'].replace({
            'Normal_Weight': 0,
            'Overweight_Level_I': 1,
            'Overweight_Level_II': 2,
            'Obesity_Type_I': 3,
            'Insufficient_Weight': 4
        }, inplace=True)
        data['family_history_with_overweight']=le.fit_transform(data['family_history_with_overweight'].values)
        data['FAVC'] = le.fit_transform(data['FAVC'].values)
        data['CAEC'] = le.fit_transform(data['CAEC'].values)
        data['SMOKE'] = le.fit_transform(data['SMOKE'].values)
        data['SCC'] = le.fit_transform(data['SCC'].values)
        data['CALC'] = le.fit_transform(data['CALC'].values)
        data['MTRANS'] = le.fit_transform(data['MTRANS'].values)
        data.loc[data['Age'] < 22 , 'Age'] = 0
        data.loc[data['Age'] >= 22, 'Age'] = 1


        label = 'y'
        positive_label = 0
        k = 10

        if numberoffeatures == 1:
            if singlefeature != 2:
                protected_group = {'Gender': 1}
            else:
                protected_group = {'Age':1}
        if numberoffeatures == 2:
            protected_group = {'Gender':1, 'Age':1}
        if numberoffeatures == 3:
            protected_group = {'Gender':1, 'Age':1, 'MTRANS':1}

            key = 'MTRANS'
            threshold = 3
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        if numberoffeatures == 4:
            protected_group = {'Gender':1, 'Age':1, 'MTRANS':1,'family_history_with_overweight':0}

            key = 'MTRANS'
            threshold = 3
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'family_history_with_overweight'
            threshold = 0.1
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        sensitive_vars = protected_group.keys()
        return data, label, positive_label, sensitive_vars, protected_group, k 

    elif(dataset == 'park'):

        data = pd.read_csv('data/park.csv')
        data.drop(['subject#', 'a', 'y', 'yhat', 'motor_UPDRS', 'total_UPDRS', 'test_time'], axis=1, inplace=True)
        data.loc[data['age']<65, 'age'] = 0
        data.loc[data['age']>=65, 'age'] = 1
        data['score_cut'].replace({
            'Mild': 0,
            'Moderate': 1,
            'Severe': 2
        }, inplace=True)
        changed_labels = data[(data['age']==1)&(data['sex']==1)&(data['score_cut']==1)].sample(n=200).index
        data.loc[changed_labels, 'score_cut'] = 0

        label = 'score_cut'
        positive_label = 0

        if numberoffeatures == 1:
            if singlefeature !=2:
                protected_group = {'age': 1}
            else:
                protected_group = {'sex':0}
        
        if numberoffeatures ==2:
            protected_group = {'age':1, 'sex':0}

        if numberoffeatures == 3:
            protected_group = {'age':1, 'sex':0, 'PPE':0}

            key = 'PPE'
            threshold = 0.14
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        if numberoffeatures == 4:

            protected_group = {'age':1, 'sex':0, 'PPE':0, 'Shimmer':1}

            key = 'PPE'
            threshold = 0.14
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'Shimmer'
            threshold = 0.02
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1



        sensitive_vars = protected_group.keys()
        k = 34
        return data, label, positive_label, sensitive_vars, protected_group, k 

    elif(dataset == 'wine'):

        red = pd.read_csv('data/winequality-red.csv', sep=';')
        red['type'] = 0
        white = pd.read_csv('data/winequality-white.csv', sep=';')
        white['type'] = 1
        data = red.append(white)
        data.drop(data[(data['quality']==3)|(data['quality']==9)|(data['quality']==8)].index, inplace=True)
        data.loc[data['alcohol'] <= 10, 'alcohol'] = 0
        data.loc[(data['alcohol'] > 10) & (data['alcohol'] != 0), 'alcohol'] = 1

        label = 'quality'
        sensitive_variables = ['alcohol', 'type']
        protected_group = {'alcohol': 0, 'type': 1}
        positive_label = 6

        if numberoffeatures == 1:
            if singlefeature != 2:
                protected_group = {'alcohol':0}
            else:
                protected_group = {'type': 1}
        
        if numberoffeatures == 2:
            protected_group = {'alcohol': 0 , 'type':1}

        if numberoffeatures == 3:
            protected_group = {'alcohol':0, 'type': 1, 'density':0 }

            key = 'density'
            threshold = 1.1
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        if numberoffeatures == 4:
            protected_group = {'alcohol':0, 'type': 1, 'density':0, 'pH':1 }

            key = 'density'
            threshold = 1.1
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

            key = 'pH'
            threshold = 3.2
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        sensitive_variables = protected_group.keys()
        k = 76
        return data, label, positive_label, sensitive_variables, protected_group, k

    else:
        print("Wrong dataset chosen. No dataset called " + str(dataset) + " found.")