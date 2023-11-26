#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
from scipy import optimize

import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 
import os


# In[3]:


df = pd.read_csv("EarthQuake.csv")
df = df.drop([3378,7512,20650]) 


# In[4]:


df.columns


# In[5]:


import matplotlib.pyplot as plt
import plotly.express as px
fig=px.density_mapbox(df,lat='Latitude',lon='Longitude',radius=1,
                      center=dict(lat=0,lon=180),zoom=1.5,mapbox_style="stamen-terrain")
fig.show(figsize = (20,25))


# In[8]:


sns.boxplot(x="Depth", y="Magnitude", hue="Type", data=df, palette="coolwarm")
plt.show()


# In[5]:


df.describe()


# In[6]:


plt.hist(df["Magnitude"],log=True)


# In[7]:


df_japan = df.query('25 < Latitude < 50 and 125 < Longitude < 150')
df_japan = df_japan.reset_index(drop=True)
df_japan["Date_Time"] = pd.to_datetime(df_japan["Date"]+" "+df_japan["Time"], format = "%m/%d/%Y %H:%M:%S")


# In[8]:


df_japan.describe()


# In[11]:


from scipy.stats import poisson
from hmmlearn import hmm

earthquakes = np.array([
    13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18,
    25, 21, 21, 14, 8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26,
    13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41, 31, 27, 35, 26,
    28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15, 22, 18, 15, 20,
    15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
    18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20,
    15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11])

fig, ax = plt.subplots()
ax.plot(earthquakes, ".-", ms=6, mfc="orange", alpha=0.7)
ax.set_xticks(range(0, earthquakes.size, 10))
ax.set_xticklabels(range(1906, 2007, 10))
ax.set_xlabel('Year')
ax.set_ylabel('Count')
fig.show()


# In[12]:


scores = list()
models = list()
for n_components in range(1, 5):
    for idx in range(10):  
        
        model = hmm.PoissonHMM(n_components=n_components, random_state=idx,
                               n_iter=10)
        model.fit(earthquakes[:, None])
        models.append(model)
        scores.append(model.score(earthquakes[:, None]))
        print(f'Converged: {model.monitor_.converged}\t\t'
              f'Score: {scores[-1]}')


model = models[np.argmax(scores)]
print(f'The best model had a score of {max(scores)} and '
      f'{model.n_components} components')

states = model.predict(earthquakes[:, None])


# In[13]:


fig, ax = plt.subplots()
ax.plot(model.lambdas_[states], ".-", ms=6, mfc="orange")
ax.plot(earthquakes)
ax.set_title('States compared to generated')
ax.set_xlabel('State')


# In[15]:


prop_per_state = model.predict_proba(earthquakes[:, None]).mean(axis=0)
bins = sorted(np.unique(earthquakes))
fig, ax = plt.subplots()
ax.hist(earthquakes, bins=bins, density=True)
ax.plot(bins, poisson.pmf(bins, model.lambdas_).T @ prop_per_state)
ax.set_title('Earthquakes_Fitted Poisson')
ax.set_xlabel('# of Earthquakes')
ax.set_ylabel('Proportion')
plt.show()


# In[9]:


def ramda(param,Tmax,Mc,Ti,Mi,T):
    a,c,p,k,mu=np.exp(param)
    return mu+((T>Ti)*k*np.exp(a*(Mi - Mc))/((T>Ti)*(T - Ti) + c)**p).sum()

def transformed_time(param,Tmax,Mc,Ti,Mi,T):
    a,c,p,k,mu=np.exp(param)    
    if abs(p-1)<0.001:
        N = mu*T + ((T>=Ti)*k*np.exp(a*(Mi - Mc))* \
                    ( (np.log((T>=Ti)*(T - Ti)+ c)**1-np.log(c)**1) \
                     +(np.log((T>=Ti)*(T - Ti)+ c)**2-np.log(c)**2)*(1-p)**1/2 \
                     +(np.log((T>=Ti)*(T - Ti)+ c)**3-np.log(c)**3)*(1-p)**2/6 \
                   )).sum() #using Taylor series at p=1
    else:
        N = mu*T + ((T>=Ti)*k*np.exp(a*(Mi - Mc))*(((T>=Ti)*(T - Ti) + c)**(1-p)-c**(1-p))/(1-p)).sum()
    return N


def calc_aic(param,Tmax,Mc,Ti,Mi):
    a,c,p,k,mu=np.exp(param)
    if abs(p-1)<0.001:
        LogL = np.array([np.log(mu+((t>Ti)*k*np.exp(a*(Mi - Mc))/((t>Ti)*(t - Ti) + c)**p).sum()) for t in Ti]).sum() \
                   - mu*Tmax - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))* \
                   ( (np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**1-np.log(c)**1) \
                    +(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**2-np.log(c)**2)*(1-p)**1/2 \
                    +(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**3-np.log(c)**3)*(1-p)**2/6 \
                   )).sum()
    else:
        LogL = np.array([np.log(mu+((t>Ti)*k*np.exp(a*(Mi - Mc))/((t>Ti)*(t - Ti) + c)**p).sum()) for t in Ti]).sum() \
                   - mu*Tmax - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))*(((Tmax>=Ti)*(Tmax - Ti) + c)**(1-p)-c**(1-p))/(1-p)).sum()
    return -2*LogL+2*5

def calc_grad(param,Tmax,Mc,Ti,Mi):
    a,c,p,k,mu=np.exp(param)
    grad=np.zeros(5)
    
    #∂/∂xΣlogλ(t,x)
    for i, t in enumerate(Ti):
        ramda=mu+((t>Ti)*k*np.exp(a*(Mi - Mc))/((t>Ti)*(t - Ti) + c)**p).sum()
        grad[0]+=((t>Ti)*k*np.exp(a*(Mi - Mc))*(Mi - Mc)                      /((t>Ti)*(t - Ti) + c)**p    ).sum()/ramda
        grad[1]+=((t>Ti)*k*np.exp(a*(Mi - Mc))*(-p)                           /((t>Ti)*(t - Ti) + c)**(p+1)).sum()/ramda
        grad[2]+=((t>Ti)*k*np.exp(a*(Mi - Mc))*np.log(1/((t>Ti)*(t - Ti) + c))/((t>Ti)*(t - Ti) + c)**p    ).sum()/ramda
        grad[3]+=((t>Ti)*1*np.exp(a*(Mi - Mc))                                /((t>Ti)*(t - Ti) + c)**p    ).sum()/ramda
        grad[4]+= 1/ramda
    

    if abs(p-1)<0.001:
        grad[0]+= - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))*(Mi - Mc)* \
                       (    (np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**1-np.log(c)**1) \
                           +(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**2-np.log(c)**2)*(1-p)**1/2 \
                           +(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**3-np.log(c)**3)*(1-p)**2/6 \
                       )).sum() 
        grad[1]+= - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))*(((Tmax>=Ti)*(Tmax - Ti) + c)**(0-p)-c**(0-p))).sum()
        grad[2]+= - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))* \
                       (   -(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**2-np.log(c)**2)/2 \
                           -(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**3-np.log(c)**3)*(1-p)/3 \
                           -(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**4-np.log(c)**4)*(1-p)**2/8 \
                       )).sum()
        grad[3]+= - ((Tmax>=Ti)*np.exp(a*(Mi - Mc))* \
                       (    (np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**1-np.log(c)**1) \
                           +(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**2-np.log(c)**2)*(1-p)**1/2 \
                           +(np.log((Tmax>=Ti)*(Tmax - Ti)+ c)**3-np.log(c)**3)*(1-p)**2/6 \
                       )).sum()
        grad[4]+= - Tmax

    else:
        dp=0.001
        pp0=(((Tmax>=Ti)*(Tmax - Ti) + c)**(1-p+dp/2)-c**(1-p+dp/2))/(1-p+dp/2)
        pp1=(((Tmax>=Ti)*(Tmax - Ti) + c)**(1-p-dp/2)-c**(1-p-dp/2))/(1-p-dp/2)
        pp = (pp1-pp0)/dp
        
        grad[0]+= - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))*(Mi - Mc)*(((Tmax>=Ti)*(Tmax - Ti) + c)**(1-p)-c**(1-p))/(1-p)).sum()
        grad[1]+= - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))*1        *(((Tmax>=Ti)*(Tmax - Ti) + c)**(0-p)-c**(0-p)      )).sum()
        grad[2]+= - ((Tmax>=Ti)*k*np.exp(a*(Mi - Mc))*pp                                                            ).sum()
        grad[3]+= - ((Tmax>=Ti)*1*np.exp(a*(Mi - Mc))*1        *(((Tmax>=Ti)*(Tmax - Ti) + c)**(1-p)-c**(1-p))/(1-p)).sum()
        grad[4]+= - Tmax

    return -2*grad*np.exp(param)


# In[10]:


Mi=df_japan["Magnitude"].values
Ti=Mi*0
for i,t in enumerate(df_japan["Date_Time"]):
    Ti[i] = (t-df_japan["Date_Time"][0]).total_seconds()/60/60/24

Mc=5.5 
Tmax=Ti.max()

a,c,p,k,mu=1.0, 0.1, 1.1, 0.1, 0.1
param_initial=np.array([a,c,p,k,mu])


# In[11]:


log_param_initial=np.log(param_initial) #log-param
args=(Tmax,Mc,Ti,Mi)

log_param_fitted=optimize.fmin_bfgs(calc_aic, log_param_initial, fprime=calc_grad, args=(Tmax,Mc,Ti,Mi),gtol=0.0001)
param_fitted = np.exp(log_param_fitted)

print("a,c,p,k,mu=",param_fitted)


# In[12]:


tmp=[ramda(log_param_fitted,Tmax,Mc,Ti,Mi,t) for t in np.arange(0,Tmax,0.1)]
fig, ax = plt.subplots(figsize=(15,5))
x = np.arange(0,Tmax,0.1)
y = np.log(tmp)
ax.set_xlabel("Time(days)")
ax.set_ylabel("log(λ(t))")
ax.plot(x, y,linewidth = 0.2)
plt.show() 


# In[13]:


tmp=[transformed_time(log_param_fitted,Tmax,Mc,Ti,Mi,t) for t in Ti]
fig, ax = plt.subplots(figsize=(10,10)) 
x = np.arange(1, 2092)
y = tmp[1:2092]
ax.set_xlabel("Number of Earthquakes")
ax.set_ylabel("Transformed Time")
ax.plot(x, y)
ax.plot(x, x)
plt.show()


# In[ ]:




