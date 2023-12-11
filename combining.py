import numpy as np
import pandas as pd
from statistics import mode
import matplotlib.pyplot as plt
#COOKING
dfK = pd.read_pickle('kitchenData.pkl')
daily_windowK= []
dfSizeK  = len(dfK)
windowLensK =[]
#print(dfSize)
k=0
while(k<dfSizeK):
    j=k
    while(True):
        if(k==dfSizeK-1):
            window = dfK.iloc[j:k]
            daily_windowK.append(window)
            k+=1
            windowLensK.append(len(window))
            break
        elif(dfK.at[k,'hour']>dfK.at[k+1,'hour']):
            window = dfK.iloc[j:k+1]
            daily_windowK.append(window)
            k+=1
            windowLensK.append(len(window))
            break
        k+=1
lenModeK = mode(windowLensK)
daily_windowK[:] = [window for window in daily_windowK if len(window) == lenModeK]
#RELAXING
dfR = pd.read_pickle('livingData.pkl')
daily_windowR= []
dfSizeR  = len(dfR)
windowLensR =[]
#print(dfSize)
r=0
while(r<dfSizeR):
    j=r
    while(True):
        if(r==dfSizeR-1):
            window = dfR.iloc[j:r]
            daily_windowR.append(window)
            r+=1
            windowLensR.append(len(window))
            break
        elif(dfR.at[r,'hour']>dfR.at[r+1,'hour']):
            window = dfR.iloc[j:r+1]
            daily_windowR.append(window)
            r+=1
            windowLensR.append(len(window))
            break
        r+=1
lenModeR = mode(windowLensR)
daily_windowR[:] = [window for window in daily_windowR if len(window) == lenModeR]
#SLEEPING
dfS = pd.read_pickle('bedData.pkl')
daily_windowS= []
dfSizeS  = len(dfS)
windowLensS =[]
#print(dfSize)
s=0
while(s<dfSizeS):
    j=s
    while(True):
        if(s==dfSizeS-1):
            window = dfS.iloc[j:s]
            daily_windowS.append(window)
            s+=1
            windowLensS.append(len(window))
            break
        elif(dfS.at[s,'hour']>dfS.at[s+1,'hour']):
            window = dfS.iloc[j:s+1]
            daily_windowS.append(window)
            s+=1
            windowLensS.append(len(window))
            break
        s+=1
lenModeS = mode(windowLensS)
daily_windowS[:] = [window for window in daily_windowS if len(window) == lenModeS]
"""
print("Kitchen days: ",len(daily_windowK))
countK=0
for colK in daily_windowK:
    length = (len(colK))
    if(length==1440):
        countK +=1
print(countK)
print("Relaxing days: ",len(daily_windowR))
countR=0
for colR in daily_windowR:
    length = (len(colR))
    if(length==1440):
        countR +=1
print(countR)

print("Sleeping days: ",len(daily_windowS))
countS=0
for colS in daily_windowS:
    length = (len(colS))
    if(length==1440):
        countS +=1
print(countS)
"""
Monday = pd.DataFrame({
    'cooking':[None]*15,
    'relaxing':[None]*15,
    'sleeping':[None]*15,
    'combined':[None]*15
})
Tuesday = pd.DataFrame({
    'cooking':[None]*15,
    'relaxing':[None]*15,
    'sleeping':[None]*15,
    'combined':[None]*15
})
Wednesday = pd.DataFrame({
    'cooking':[None]*15,
    'relaxing':[None]*15,
    'sleeping':[None]*15,
    'combined':[None]*15
})
Thursday = pd.DataFrame({
    'cooking':[None]*15,
    'relaxing':[None]*15,
    'sleeping':[None]*15,
    'combined':[None]*15
})
Friday = pd.DataFrame({
    'cooking':[None]*15,
    'relaxing':[None]*15,
    'sleeping':[None]*15,
    'combined':[None]*15
})
Saturday = pd.DataFrame({
    'cooking':[None]*15,
    'relaxing':[None]*15,
    'sleeping':[None]*15,
    'combined':[None]*15
})
Sunday = pd.DataFrame({
    'cooking':[None]*15,
    'relaxing':[None]*15,
    'sleeping':[None]*15,
    'combined':[None]*15
})
sunK,monK,tueK,wedK,thuK,friK,satK = 0,0,0,0,0,0,0
for i in range(len(daily_windowK)):
    if daily_windowK[i]["day_of_week"].nunique()==1:
        dayK = daily_windowK[i]["day_of_week"].unique()[0]
        if(dayK==0 and sunK<15):
            Sunday['cooking'][sunK] = daily_windowK[i]["cooking_label"]
            sunK+=1
        if(dayK==1 and monK<15):
            Monday['cooking'][monK] = daily_windowK[i]["cooking_label"]
            monK+=1
        if(dayK==2 and tueK <15):
            Tuesday['cooking'][tueK] = daily_windowK[i]["cooking_label"]
            tueK+=1
        if(dayK==3 and wedK<15):
            Wednesday['cooking'][wedK] = daily_windowK[i]["cooking_label"]
            wedK+=1
        if(dayK==4 and thuK <15):
            Thursday['cooking'][thuK] = daily_windowK[i]["cooking_label"]
            thuK+=1
        if(dayK==5 and friK<15):
            Friday['cooking'][friK] = daily_windowK[i]["cooking_label"]
            friK+=1
        if(dayK==6 and satK<15):
            Saturday['cooking'][satK] = daily_windowK[i]["cooking_label"]
            satK+=1

sunR,monR,tueR,wedR,thuR,friR,satR = 0,0,0,0,0,0,0
for i in range(len(daily_windowR)):
    if daily_windowR[i]["day_of_week"].nunique()==1:
        dayR = daily_windowR[i]["day_of_week"].unique()[0]
        if(dayR==0 and sunR<15):
            Sunday['relaxing'][sunR] = daily_windowR[i]["living_label"]
            sunR+=1
        if(dayR==1 and monR<15):
            Monday['relaxing'][monR] = daily_windowR[i]["living_label"]
            monR+=1
        if(dayR==2 and tueR <15):
            Tuesday['relaxing'][tueR] = daily_windowR[i]["living_label"]
            tueR+=1
        if(dayR==3 and wedR<15):
            Wednesday['relaxing'][wedR] = daily_windowR[i]["living_label"]
            wedR+=1
        if(dayR==4 and thuR<15):
            Thursday['relaxing'][thuR] = daily_windowR[i]["living_label"]
            thuR+=1
        if(dayR==5 and friR<15):
            Friday['relaxing'][friR] = daily_windowR[i]["living_label"]
            friR+=1
        if(dayR==6 and satR<15):
            Saturday['relaxing'][satR] = daily_windowR[i]["living_label"]
            satR+=1
sunS,monS,tueS,wedS,thuS,friS,satS = 0,0,0,0,0,0,0
for i in range(len(daily_windowS)):
    if daily_windowS[i]["day_of_week"].nunique()==1:
        dayS = daily_windowS[i]["day_of_week"].unique()[0]
        if(dayS==0 and sunS<15):
            Sunday['sleeping'][sunS] = daily_windowS[i]["bed_label"]
            sunS+=1
        if(dayS==1 and monS<15):
            Monday['sleeping'][monS] = daily_windowS[i]["bed_label"]
            monS+=1
        if(dayS==2 and tueS<15):
            Tuesday['sleeping'][tueS] = daily_windowS[i]["bed_label"]
            tueS+=1
        if(dayS==3) and wedS<15:
            Wednesday['sleeping'][wedS] = daily_windowS[i]["bed_label"]
            wedS+=1
        if(dayS==4 and thuS<15):
            Thursday['sleeping'][thuS] = daily_windowS[i]["bed_label"]
            thuS+=1
        if(dayS==5 and friS<15):
            Friday['sleeping'][friS] = daily_windowS[i]["bed_label"]
            friS+=1
        if(dayS==6 and satS <15):
            Saturday['sleeping'][satS] = daily_windowS[i]["bed_label"]
            satS+=1
print(sunK,sunR,sunS)
print(monK,monR,monS)
print(tueK,tueR,tueS)
print(wedK,wedR,wedS)
print(thuK,thuR,thuS)
print(friK,friR,friS)
print(satK,satR,satS)
for i in range(len(Sunday)):
    Sunday["combined"][i]= Sunday["cooking"][i].add(Sunday["relaxing"][i],fill_value =0).add(Sunday["sleeping"][i],fill_value=0)
for i in range(len(Monday)):
    #Monday["combined"][i]= Monday["cooking"][i]+Monday["relaxing"][i]+Monday["sleeping"][i]
    Monday["combined"][i]= Monday["cooking"][i].add(Monday["relaxing"][i],fill_value =0).add(Monday["sleeping"][i],fill_value=0)
for i in range(len(Tuesday)):
    #Tuesday["combined"][i]= Tuesday["cooking"][i]+Tuesday["relaxing"][i]+Tuesday["sleeping"][i]
    Tuesday["combined"][i]= Tuesday["cooking"][i].add(Tuesday["relaxing"][i],fill_value =0).add(Tuesday["sleeping"][i],fill_value=0)
for i in range(len(Wednesday)):
    #Wednesday["combined"][i]= Wednesday["cooking"][i]+Wednesday["relaxing"][i]+Wednesday["sleeping"][i]
    Wednesday["combined"][i]= Wednesday["cooking"][i].add(Wednesday["relaxing"][i],fill_value =0).add(Wednesday["sleeping"][i],fill_value=0)
for i in range(len(Thursday)):
    #Thursday["combined"][i]= Thursday["cooking"][i]+Thursday["relaxing"][i]+Thursday["sleeping"][i]
    Thursday["combined"][i]= Thursday["cooking"][i].add(Thursday["relaxing"][i],fill_value =0).add(Thursday["sleeping"][i],fill_value=0)
for i in range(len(Friday)):
    #Friday["combined"][i]= Friday["cooking"][i]+Friday["relaxing"][i]+Friday["sleeping"][i]
    Friday["combined"][i]= Friday["cooking"][i].add(Friday["relaxing"][i],fill_value =0).add(Friday["sleeping"][i],fill_value=0)
for i in range(len(Saturday)):
    #Saturday["combined"][i]= Saturday["cooking"][i]+Saturday["relaxing"][i]+Saturday["sleeping"][i]
    Saturday["combined"][i]= Saturday["cooking"][i].add(Saturday["relaxing"][i],fill_value =0).add(Saturday["sleeping"][i],fill_value=0)
# print(Tuesday.iloc[0]['relaxing'])
# print("cooking")
# print(Tuesday.iloc[0]['cooking'])
# print("sleeping")
# print(Tuesday.iloc[0]['sleeping'])
# print("combined")
w = (Tuesday.iloc[0]['combined'])
# print("Min")
# print(Tuesday.iloc[0]['combined'].min())
# print("Max")
# print(Tuesday.iloc[0]['combined'].max())
sun,mon,tue,wed,thu,fri,sat = 0,0,0,0,0,0,0
#series = pd.DataFrame(columns= ['cooking','relaxing','sleeping','combined'])
series = pd.DataFrame({'combined':[100]})
#print(series)
for i in range(105):
    #print("Day: ",i)
    if (i%7==0):
        #series.loc[len(series)] = Tuesday.iloc[tue]
        dataTue = Tuesday.iloc[tue]['combined'].head(1440)
        tueSeries = pd.DataFrame(dataTue.tolist(), columns=['combined'])
        #print("Tue: ",len(tueSeries))
        ##print(tueSeries)
        series = pd.concat([series,tueSeries],ignore_index=True)
        tue+=1
    if (i%7==1):
        #series.loc[len(series)] = Wednesday.iloc[wed]
        dataWed = Wednesday.iloc[wed]['combined'].head(1440)
        wedSeries = pd.DataFrame(dataWed.tolist(), columns=['combined'])
        #print("Wed: ",len(wedSeries))
        series = pd.concat([series,wedSeries],ignore_index=True)
        wed+=1
    if (i%7==2):
        #series.loc[len(series)] = Thursday.iloc[thu]
        dataThu = Thursday.iloc[thu]['combined'].head(1440)
        thuSeries = pd.DataFrame(dataThu.tolist(), columns=['combined'])
        #print("Thu: ",len(thuSeries))
        series = pd.concat([series,thuSeries],ignore_index=True)
        thu+=1
    if (i%7==3):
        #series.loc[len(series)] = Friday.iloc[fri]
        dataFri = Friday.iloc[fri]['combined'].head(1440)
        friSeries = pd.DataFrame(dataFri.tolist(), columns=['combined'])
        #print("Fri: ",len(friSeries))
        series= pd.concat([series,friSeries],ignore_index=True)
        fri+=1
    if (i%7==4):
        #series.loc[len(series)] = Saturday.iloc[sat]
        dataSat = Saturday.iloc[sat]['combined'].head(1440)
        satSeries = pd.DataFrame(dataSat.tolist(), columns=['combined'])
        #print("Sat: ",len(satSeries))
        series = pd.concat([series,satSeries],ignore_index=True)
        sat+=1
    if (i%7==5):
        #series.loc[len(series)] = Sunday.iloc[sun]
        dataSun = Sunday.iloc[sun]['combined'].head(1440)
        sunSeries = pd.DataFrame(dataSun.tolist(), columns=['combined'])
        #print("Sun: ",len(sunSeries))
        series= pd.concat([series,sunSeries],ignore_index=True)
        sun+=1
    if (i%7==6):
        #series.loc[len(series)] = Monday.iloc[mon]
        dataMon = Monday.iloc[mon]['combined'].head(1440)
        monSeries = pd.DataFrame(dataMon.tolist(), columns=['combined'])
        #print("Mon: ",len(monSeries))
        series= pd.concat([series,monSeries],ignore_index=True)
        #series['combined'] = pd.concat([temp['combined'],Monday.iloc[mon]['combined']],ignore_index=True)
        mon+=1
    #print("series:",len(series))
#print(len(series))
print("series")
series['combined'] = series['combined'].iloc[1:]
series.to_pickle("theSeries.pkl")
#print(series)
#print(series.min())
#print(series.max())

# plt.plot(series['combined'].iloc[1:10081])
# plt.plot(series['combined'].iloc[10081:20161])
# plt.plot(series['combined'].iloc[20161:30241])
# plt.plot(series['combined'].iloc[30241:40321])
# plt.show()