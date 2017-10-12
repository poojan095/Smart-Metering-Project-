
# coding: utf-8

# In[3]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the Census dataset
data = pd.read_csv("Electricity consumption data1.csv")

# Success - Display the first record
display(data.head(n=10))



# # Plotting Data

# In[4]:

import matplotlib.pyplot as plt
import csv
import datetime


Y = data['102']
date = data['Time']
fig, ax = plt.subplots()

ax.plot_date(date[:3000],Y[:3000],markersize=.1,linewidth=1, linestyle="-")
plt.gcf().autofmt_xdate()
ax.set_xlim([datetime.date(2017, 3, 12), datetime.date(2017, 3, 13)])
ax.set_ylim([0, 0.20])
#plt.rcParams["figure.figsize"] = [20,9]

plt.show()


# # Average

# In[5]:

average = []
n_records = len(data)
for i in range(0,100):  
    count = 0
    x = 0
    for key in data:
        if count == 0:
            count = 1 
        else:
            x = x + data[key][i]

    average.append(x/211)
    
fig, ax1 = plt.subplots()

Y = data['104'];
date = data['Time']
ax1.plot_date(date[:100],average[:100],markersize=.1,linewidth=1, linestyle="-", color = 'r')

ax1.plot_date(date[:100],Y[:100],markersize=.1,linewidth=1, linestyle="-")
ax1.set_xlim([datetime.date(2017, 3, 9), datetime.date(2017, 3, 10)])
ax1.set_ylim([0, 0.20])
#plt.rcParams["figure.figsize"] = [20,9]

plt.gcf().autofmt_xdate()

plt.show()


# # Predictions

# In[6]:


predictions = []
for i in range(1,95):
    total = 0
    
    for j in range(i-1,i+2):
        total = total + Y[j]
    
    if Y[i] > 0.04 and  (total/4) >= 0.03: #if the apartment is using more than 0.04
        predictions.append(1)
    

    elif Y[i] - Y[i-1] > 0.04 or Y[i] - Y[i-1] < -0.04 or Y[i] - Y[i+1] > 0.04 or Y[i] - Y[i+1] < -0.04:
        predictions.append(1)
            
    else:
        predictions.append(0)

fig, ax1 = plt.subplots()

ax1.plot_date(date[:95],Y[:95],markersize=.1,linewidth=1, linestyle="-")
plt.gcf().autofmt_xdate()
ax1.set_xlim([datetime.date(2017, 3, 9), datetime.date(2017, 3, 10)])
ax1.set_ylim([0, 0.30])

ax2 = ax1.twinx()
ax2.plot_date(date[:94],predictions,markersize=.1,linewidth=1, linestyle="-", color = 'r')
ax2.set_xlim([datetime.date(2017, 3, 9), datetime.date(2017, 3, 10)])

ax2.set_ylim([-0.003, 1.01])

#plt.rcParams["figure.figsize"] = [20,9]

plt.show()


# # Hourly Data

# In[7]:

n = n_records
hourly = []
for i in range(0,n,4):
    if i <= (n-4):
        total = data['101'][i] + data['101'][i+1] + data['101'][i+2] + data['101'][i+3]

        hourly.append(total/4)
    else: 
        break
hours = []
for i in range(0,n,4):
    hours.append(data['Time'][i])
    
fig, ax1 = plt.subplots()

    
ax1.plot_date(hours[:1000],hourly[:1000],markersize=.1,linewidth=1, linestyle="-",color = 'r')
ax1.set_ylim([0, 0.20])


ax1.plot_date(date[:3000],data['101'][:3000],markersize=.1,linewidth=1, linestyle="-")
ax1.set_xlim([datetime.date(2017, 3, 9), datetime.date(2017, 3, 10)])
#plt.rcParams["figure.figsize"] = [20,9]

plt.gcf().autofmt_xdate()
plt.show()



# In[8]:

from scipy.signal import butter, filtfilt
import numpy as np
x = 1  # number of days
def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=True)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

rawdata = data['101'][:96*x]
signal = rawdata
fs = 2

cutoff = .5


order = 1
conditioned_signal = butter_highpass_filter(signal, cutoff, fs, order)

T = 48*x
nsamples = T * fs
t = np.linspace(0, T, nsamples, endpoint=False)


fig, ax1 = plt.subplots()


ax1.plot(t,conditioned_signal,markersize=.1,linewidth=1, linestyle="-", label='Corrected signal')
ax1.plot(t,rawdata,markersize=.1,linewidth=1, linestyle="-",color = 'r', label='Noisy signal')

#ax1.set_xlim([datetime.date(2017, 3, 9), datetime.date(2017, 3, 10)])
plt.legend(loc='upper left')





# In[11]:

count = 0
X = 0

for item in Y:
    count = count + 1
    if count < 120:
        X = X + item * 1.45
    else:
        X = X + item * 7.6
        
print X


# In[10]:

X = 0
for item in Y:
        X = X + item * 9.627
print X
        


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



