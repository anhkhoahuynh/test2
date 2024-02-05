import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from io import StringIO

df = pd.read_csv('clone7 (1).csv')
X = pd.concat([df.iloc[:,1], df.iloc[:,2], df.iloc[:,3],df.iloc[:,4],df.iloc[:,5],df.iloc[:,6],df.iloc[:,7]], axis=1)

#Remove outliner - TDS, HCO3
Q1_TDS = X[['TDS']].quantile(0.25)
Q3_TDS = X[['TDS']].quantile(0.75)
IQR_TDS = Q3_TDS - Q1_TDS
lower_TDS = Q1_TDS - 1.5*IQR_TDS
upper_TDS = Q3_TDS + 1.5*IQR_TDS
Q1_bicar = X[['HCO3']].quantile(0.25)
Q3_bicar = X[['HCO3']].quantile(0.75)
IQR_bicar = Q3_bicar - Q1_bicar
lower_bicar = Q1_bicar - 1.5*IQR_bicar
upper_bicar = Q3_bicar + 1.5*IQR_bicar
upper_array_bicar = np.where(X[['HCO3']] >= upper_bicar)[0]
lower_array_bicar = np.where(X[['HCO3']] <= lower_bicar)[0]
upper_array_TDS = np.where(X[['TDS']] >= upper_TDS)[0]
lower_array_TDS = np.where(X[['TDS']] <= lower_TDS)[0]
X.drop(index=upper_array_bicar, inplace=True)
X.drop(index=lower_array_bicar, inplace=True)
X.drop(index=upper_array_TDS, inplace=True)
X.drop(index=lower_array_TDS, inplace=True)
X.reset_index(drop=True, inplace = True)

#Remove outliner - Ca
Q1_Ca = X[['Ca']].quantile(0.25)
Q3_Ca = X[['Ca']].quantile(0.75)
IQR_Ca = Q3_Ca - Q1_Ca
lower_Ca = Q1_Ca - 1.5*IQR_Ca
upper_Ca = Q3_Ca + 1.5*IQR_Ca
upper_array_Ca = np.where(X[['Ca']] >= upper_Ca)[0]
lower_array_Ca = np.where(X[['Ca']] <= lower_Ca)[0]
X.drop(index=upper_array_Ca, inplace=True)
X.drop(index=lower_array_Ca, inplace=True)
X.reset_index(drop=True, inplace = True)

#Remove outliner - Mg
Q1_Mg = X[['Mg']].quantile(0.25)
Q3_Mg = X[['Mg']].quantile(0.75)
IQR_Mg = Q3_Mg - Q1_Mg
lower_Mg = Q1_Mg - 1.5*IQR_Mg
upper_Mg = Q3_Mg + 1.5*IQR_Mg
upper_array_Mg = np.where(X[['Mg']] > upper_Mg)[0]
lower_array_Mg = np.where(X[['Mg']] <= lower_Mg)[0]
X.drop(index=upper_array_Mg, inplace=True)
X.drop(index=lower_array_Mg, inplace=True)
X.reset_index(drop=True, inplace = True)

#Remove outliner - pH
Q1_pH = X[['pH']].quantile(0.25)
Q3_pH = X[['pH']].quantile(0.75)
IQR_pH = Q3_pH - Q1_pH
lower_pH = Q1_pH - 1.5*IQR_pH
upper_pH = Q3_pH + 1.5*IQR_pH
upper_array_pH = np.where(X[['pH']] >= upper_pH)[0]
lower_array_pH = np.where(X[['pH']] <= lower_pH)[0]
X.drop(index=upper_array_pH, inplace=True)
X.drop(index=lower_array_pH, inplace=True)
X.reset_index(drop=True, inplace = True)

#Check outliner
plt.figure(figsize=(20,5))
plt.subplot(1,5,1)
sns.boxplot(X.HCO3)
plt.subplot(1,5,2)
sns.boxplot(X.Ca)
plt.subplot(1,5,3)
sns.boxplot(X.Mg)
plt.subplot(1,5,4)
sns.boxplot(X.pH)
plt.subplot(1,5,5)
sns.boxplot(X.TDS)
plt.show()
# kiểm tra distance
plt.figure(figsize=(20,5))
plt.subplot(1,5,1)
sns.histplot(X.HCO3)
plt.subplot(1,5,2)
sns.histplot(X.Ca)
plt.subplot(1,5,3)
sns.histplot(X.Mg)
plt.subplot(1,5,4)
sns.histplot(X.pH)
plt.subplot(1,5,5)
sns.histplot(X.TDS)
plt.show()

# Running the Shapiro-Wilk Test on Normal Data
from scipy.stats import shapiro
def shapiro_test(data, alpha = 0.05):
    stat, p = shapiro(data)
    if p > alpha:
        print('Shapiro-Wilk test: Gaussian, p-value = ', p)
    else:
        print('Shapiro-Wilk test: Not Gaussian, p-value = ', p)
print("Ca:")
shapiro_test(X[['Ca']])
print("\nMg:")
shapiro_test(X[['Mg']])
print("\nHCO3:")
shapiro_test(X[['HCO3']])
print("\nTDS:")
shapiro_test(X[['TDS']])
print("\npH:")
shapiro_test(X[['pH']])



from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_over = X[['HCO3','pH','TDS','Ca','Mg']]
y_over = X[['Precipitation']]
X_over, y_over = oversample.fit_resample(X_over, y_over)

target = pd.DataFrame(y_over,columns=['Precipitation'])
df_over = pd.concat([X_over,target],axis=1)
#chỉ mở khi muốn over sampling lại
'''
from google.colab import drive
drive.mount('drive')
df_over.to_csv('/content/drive/My Drive/Colab_Notebooks/Random_forest_deployed/after_over_sampling.csv')
'''

df_over = pd.read_csv('after_over_sampling.csv')
df_over = pd.concat([df_over.iloc[:,1], df_over.iloc[:,2],df_over.iloc[:,3],df_over.iloc[:,4],df_over.iloc[:,5],df_over.iloc[:,6]], axis=1)

from scipy.stats.stats import pearsonr
pearsonr(df_over['pH'], df_over['HCO3'])

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = MinMaxScaler()
df_before_scale = df_over[['HCO3','pH','TDS','Ca','Mg']]
df_after_scale = scaler.fit_transform(df_before_scale)
print(df_after_scale)
df_after_scale = pd.DataFrame(df_after_scale, columns=['rb_HCO3','rb_pH','rb_TDS','rb_Ca','rb_Mg'])
df_after_scale = pd.concat([df_over.reset_index(drop=True),df_after_scale],axis=1)
df_after_scale = df_after_scale.drop(columns=['HCO3','pH','TDS','Ca','Mg'])
df_after_scale = df_after_scale.reindex(columns=['rb_HCO3','rb_pH','rb_TDS','rb_Ca','rb_Mg','Precipitation'])
df_after_scale= df_after_scale.dropna()

# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_after_scale[['rb_HCO3','rb_Mg','rb_Ca','rb_pH','rb_TDS']],df_after_scale[['Precipitation']], test_size=0.2, random_state=0)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100,  oob_score=True, random_state=42)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
oob_score1 =  clf.oob_score_
print(X_train)
print(type(X_train))
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
print(f'OOB score: {oob_score1:.3f}')
feature_imp = pd.Series(clf.feature_importances_, index = X_train.columns).sort_values(ascending = False)
print(feature_imp)

#vẽ importance
features = X_train.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

from flask import Flask, request, render_template
from flask.templating import render_template
import numpy as np


# create app

app = Flask(__name__,template_folder='./templates')

import pickle
model_load = pickle.load(open('rung_ngau_nhien.pkl','rb'))


def scale_mau(mau):
    if (mau[0] > df_before_scale.HCO3.max() or mau[0] < df_before_scale.HCO3.min()):
        print('bạn đã nhập điểm ngoại lai HCO3, khoảng HCO3 là:', df_before_scale.HCO3.min(),'-',df_before_scale.HCO3.max())
    else:
      if (mau[1]  > df_before_scale.Mg.max() or mau[1] < df_before_scale.Mg.min()):
        print('bạn đã nhập điểm ngoại lai Mg, khoảng Mg là:', df_before_scale.Mg.min(),'-',df_before_scale.Mg.max())
      else:
        if (mau[2]  > df_before_scale.Ca.max() or mau[2] < df_before_scale.Ca.min()):
          print('bạn đã nhập điểm ngoại lai Ca, khoảng Ca là:', df_before_scale.Ca.min(),'-',df_before_scale.Ca.max())
        else:
          if (mau[3]  > df_before_scale.pH.max() or mau[3] < df_before_scale.pH.min()):
            print('bạn đã nhập điểm ngoại lai pH, khoảng pH là:', df_before_scale.pH.min(),'-',df_before_scale.pH.max())
          else:
            if (mau[4]  > df_before_scale.TDS.max() or mau[4] < df_before_scale.TDS.min()):
              print('bạn đã nhập điểm ngoại lai TDS, khoảng TDS là:', df_before_scale.TDS.min(),'-',df_before_scale.TDS.max())
            else:
              HCO3_afterscaling = (mau[0] - df_before_scale.HCO3.min())/(df_before_scale.HCO3.max()-df_before_scale.HCO3.min())
              Mg_afterscaling = (mau[1] - df_before_scale.Mg.min())/(df_before_scale.Mg.max()-df_before_scale.Mg.min())
              Ca_afterscaling = (mau[2] - df_before_scale.Ca.min())/(df_before_scale.Ca.max()-df_before_scale.Ca.min())
              pH_afterscaling = (mau[3] - df_before_scale.pH.min())/(df_before_scale.pH.max()-df_before_scale.pH.min())
              TDS_afterscaling = (mau[4] - df_before_scale.TDS.min())/(df_before_scale.TDS.max()-df_before_scale.TDS.min())
              mau_afterscaling = np.array([HCO3_afterscaling, Mg_afterscaling, Ca_afterscaling, pH_afterscaling, TDS_afterscaling])
              mau_afterscaling = mau_afterscaling.reshape(1,-1)
              #mau_afterscaling = pd.DataFrame(mau_afterscaling, columns=['rb_HCO3', 'rb_Mg','rb_Ca','rb_pH','rb_TDS'])
              return mau_afterscaling


@app.route('/')
def home():
  return render_template('index.html')

if __name__ == '__main__':
  app.run()


@app.route('/getprediction',methods=['POST'])

def getprediction():
  #nhập theo thứ tự HCO3, Mg, Ca, pH, TDS
  mau1 = np.array([450, 0.5, 1.2, 7.9, 375])
  mau1_after = scale_mau(mau1)
  #driven code
  a = model_load.predict(mau1_after)
  b = model_load.predict_proba(mau1_after)
  c = mau1_after
  input = [float(x) for x in request.form.values()]
  final_input = np.array(input)
  final_input_scale = np.array(scale_mau(final_input))
  prediction = model_load.predict(final_input_scale.reshape(1,-1))
  tyle = model_load.predict_proba(final_input_scale.reshape(1,-1))
  d = final_input_scale
  ketluan = " "
  if prediction == 1:
    ketluan = "Precipitation"
  else:
    ketluan = "No precipitation"

  return render_template('index.html',
                         raw_HCO3 = '{}'.format(input[0]),
                         raw_Mg = '{}'.format(input[1]),
                         raw_Ca = '{}'.format(input[2]),
                         raw_pH = '{}'.format(input[3]),
                         raw_TDS = '{}'.format(input[4]),
                         scale_HCO3 = '{}'.format(final_input_scale[0,0]),
                         scale_Mg = '{}'.format(final_input_scale[0,1]),
                         scale_Ca = '{}'.format(final_input_scale[0,2]),
                        scale_pH = '{}'.format(final_input_scale[0,3]),
                         scale_TDS = '{}'.format(final_input_scale[0,4]),
                         ketluan = '{}'.format(ketluan),
                         tyle = '{}/{}'.format(tyle[0,0], tyle[0,1]),
                         a= '{}'.format(a),
                         b= '{}/{}'.format(b[0,0], b[0,1]),
                         c= '{}'.format(c),
                         d= '{}'.format(d),
                        e= '{}'.format(final_input_scale)
                         )
app.run()
if __name__ == '__main__':
  app.run()
