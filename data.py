import numpy as np
import pandas as pd

df = pd.read_csv('py4ai-score.csv')

# -----------------------------------------------------CLEAN DATA-------------------------------------------

df['REG-MC4AI'].fillna('N',inplace=True)
for i in df.columns[:-1]:
    df[i].fillna(0,inplace=True)
 
#print(df)
#print(df.isnull().sum())

#------------------------------------------------ADD 'CLASS-GROUP' & 'S-AVG' & 'RESULT' COL -------------------------------------
def class_group(x):
  class_list = ['CV','CT','CA','CL','CH','TH','SN','CSD','CTIN','CTRN']
  to_group = ['Chuyên Văn','Chuyên Toán','Chuyên Anh','Chuyên Lí','Chuyên Hóa','Tích hợp','Song ngữ','Sử Địa','Chuyên Tin','Trung Nhật']
  for i in range(len(class_list)):
    if class_list[i] in x:
      if class_list[i]=='CT':
         if x[2:]=='CTIN':
            return 'Chuyên Tin'
         elif x[2:]=='CTRN':
            return 'Trung Nhật'
         else:
            return 'Chuyên Toán'
      else:
         return to_group[i]

df['CLASS-GROUP'] = df['CLASS'].apply(class_group)
df['CLASS-GROUP'].fillna('Khác',inplace=True)

df['S-AVG'] = round(df[['S1','S2','S3','S4','S5','S6','S7','S8','S9']].mean(axis=1),1)

def result(row):
   if row['GPA'] >=6.0:
      return 'Đ'
   else:
      return 'KĐ'
   
df['RESULT'] = df.apply(result,axis=1)

print(df)



