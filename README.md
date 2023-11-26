# Earthquake-predictor
Predicting earthquake occurrence

	Statistical Analysis and Probability in Earthquake Prediction


Introduction
Earthquakes are natural disasters that can have devastating consequences and causing widespread damage and loss of life. The ability to predict their occurrence and understand the underlying patterns is vital for disaster preparedness. Through comprehensive data analysis and the application of statistical techniques, we want to address the following question: Can historical earthquake data serve as a basis for predicting the likelihood or occurrence of future earthquakes?
The study of earthquakes and predicting them is our purpose of research here which is aimed at improving our understanding of seismic activity and enhancing early warning systems if possible. In this project, we delve into the realm of statistical analysis and probability to explore the relationship between previous earthquakes and upcoming ones while using our statistical knowledge and getting a little bit of help from our coding skills (mostly Python).


Data Analysis

Before going into deep analysis and creating our model let’s take a look at our dataset. First of all, we want to see what general parameters are available in our dataset so that we can choose the ones that are useful for us.

 

We can see we have a lot of numeric and non-numeric parameters in our dataset and probably we are only going to use some of them in our final model.
Now, let’s see how this dataset looks like. Note that we will use Logarithmic scale since when studying natural phenomenon like earthquakes, it is highly recommended not to use a linear scale if you want to see the intrinsic pattern that lies within the numbers.
 
Thanks to the logarithmic scale, we can clearly see that low magnitude earthquakes happen more frequently. In the next step we will see on a map of the world which areas on earth are more in danger of getting hit by an earthquake:
 
We can see that in the lower right of the map above, the density of earthquake occurrence is really high (We will use this map later in this project).
Now, let’s take a look at some statistical indexes of our dataset using “ df.describe()” command:
 
We can see some useful statistical numbers that describe our dataset like its min and maximum value, its median, average and standard deviation for each of its features, …
For example, looking at the ‘Magnitude’ column, we can see that the average magnitude of the earthquakes in the dataset is approximately 5.88, with a standard deviation of about 0.42. The smallest recorded magnitude is 5.5, and the largest is 9.1. The median magnitude (the value that divides the number of data in half) is 5.7.
Now that we get some information about our dataset and how it actually looks like, let’s make a box plot that shows the distribution of earthquake magnitudes at different depths, grouped by the type of seismic event. We will make such useful plot using seaborn library of Python.
 
The x-axis represents the depth of the earthquakes while the y-axis represents the magnitude of the earthquakes. Each box in the plot represents the interquartile range (IQR) of earthquake magnitudes at a certain depth, for a certain type of seismic event. (Note that IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile) of the data. In other words, the box contains the middle 50% of the data). Also the line inside each box represents the median magnitude (the 50th percentile) for that type of seismic event at that depth and the “whiskers” extending from the box represent the range of the data within 1.5 times the IQR and any data points outside this range are considered outliers. The colors of the boxes represent different types of seismic events, as specified by the hue="Type" argument.
Note that since the dataset is way too large, we can’t see the Type effect in our plot even if we use a bigger platform like:
 
In order to see the effect of the type we will take a sample of 0.5% of our dataset to see the full power of this plot:
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("EarthQuake.csv")
df = df.drop([3378,7512,20650]) 
# Sample 0.5% of the data
df_sample = df.sample(frac=0.005, random_state=1)
sns.boxplot(x="Depth", y="Magnitude", hue="Type", data=df_sample, palette="coolwarm")
plt.show()
 
This plot helps us understand the relationship between the depth and magnitude of different types of seismic events. For example, by running this code multiple times and using different samples, we see that certain types of events tend to occur at certain depths, or that the magnitude of events varies with depth. More specifically we can see that: The depth of earthquakes can vary greatly, from very shallow to very deep. Shallow earthquakes (0-70 km deep) are more common and often more destructive, while intermediate (70-300 km) and deep (300-700 km) earthquakes occur less frequently. Explosions would typically occur at relatively shallow depths and their magnitude far less than earthquakes or nuclear kind. Nuclear explosions usually occur at relatively shallow depths. However, we can see some nuclear tests have been conducted underground. The magnitude of a nuclear explosion looks to be very high, reflecting the enormous amount of energy released. Rock bursts usually seem to be several kilometers below the surface and their magnitude is lower than a natural earthquake of similar depth.
Now we are going to fit a distribution model on our dataset from http://earthquake.usgs.gov  
We are going to investigate large (7+) earthquakes that happened between 1906 and 2006.
 
We are going to fit a Poisson model to model this data. The reason behind selecting this distribution is:
Discreteness: The Poisson distribution is discrete like our data and earthquakes are distinct events that can be counted.
Independence: The Poisson distribution assumes that events occur independently. This is a reasonable assumption for earthquakes, as the occurrence of one earthquake does not directly influence the occurrence of another.
Constant rate: The Poisson distribution assumes that events occur at a constant average rate. While the rate of earthquakes may vary over long periods due to tectonic processes, it could be reasonably constant over short time scales.
Rare events: The Poisson distribution is often used to model rare events, and large earthquakes (magnitude 7+) are relatively rare.
Simple to model: The Poisson distribution is defined by just one parameter (the rate parameter), which simplifies the modeling process.
Now let’s find the best model based on the maximum log-likelihood of the observed data given the model:
 
So we are fitting a Poisson Model to the earthquake data for different numbers of hidden states, selecting the best model based on the data likelihood and then predicting the most likely sequence of hidden states given the observed data. Now we will plot the waiting times from our most likely series of states of earthquake activity with the earthquake data. As we can see, the model with the maximum likelihood had different states which may reflect times of varying earthquake danger.
 
Lastly, let’s examine the distribution of earthquakes in relation to our waiting time parameter values. Clearly our model aligns well with the distribution and effectively mirroring the results from the reference.
 
And we have completed our Poisson model on earthquake data. Now let’s return back to the map of the occurrence of earthquakes around the world provided at the beginning of this report.

By a simple web search (if our Geography is not good enough to know this from the provided map) we will know that Japan is one of those countries that is getting the most earthquakes, so let’s investigate this country even more but in order to make ourselves even more certain, let’s make a density mapbox plot of earthquake locations:

 
And if we search for Japan’s location on google map, we will see that its Latitude is between 25 and 50 and its Longitude is between 125 and 150. Let’s investigate Japan even more. To analyze a sequence of earthquakes (including occurrence date and magnitude), the Epidemic Type Aftershock Sequence (ETAS) model (referenced in Ogata, 1988, 1999) is beneficial. The ETAS model is a point process model that incorporates the Omori-Utsu law and a branching process where each earthquake has the potential to generate its own aftershocks. As a result, the seismicity rate at time t is calculated by adding the effects of all previous earthquakes and the background seismicity rate μ. 
 
In the equation, c and p are parameters in the Omori-Utsu law, while K and α regulate the aftershock productivity by a mainshock and its sensitivity to magnitude, respectively. These five parameters, μ, c, p, K and α are determined by Maximum Likelihood Estimation (MLE). In the ETAS model, the log likelihood can be expressed as follows:
 
Now that we are done with theory, let’s go to Python implementation. First, let’s filter the data for earthquakes occurred in Japan:
 
Now let’s get the ETAS parameters:
 
Now, let’s plot our Lambda function and transformed time(Log):
 
And finally plotting transformed time we get:
 
This plot shows the relationship between the number of earthquakes and the transformed time according to the fitted ETAS model. These plots help us understand how well the ETAS model fits the earthquake data. The model is a good fit since the seismicity rate plot closely match the distribution of earthquakes over time and the transformed time plot closely follow the diagonal line.
For more investigation, let’s fit a “RandomForest” model on our data as well.
 
Let’s change the number of estimators and get the best fit:
 

 
Note that the R2 score is a statistical measure that represents the proportion of the variance for a dependent variable that’s explained by an independent variable or variables in a regression model. 
The result we have is approximately 0.397, which means that about 39.7% of the variance in the test data can be explained by your model. This is a relatively low score, suggesting that the model might not be a great fit to the data unlike the Poisson model we previously fitted.


Conclusion
The analysis of earthquake data using statistical methods and probability models has provided valuable insights. The dataset, which includes a variety of parameters, was explored and key features were selected for further analysis. The use of a logarithmic scale revealed the intrinsic pattern of earthquake occurrences, showing that low magnitude earthquakes happen more frequently. The geographical distribution of earthquakes was visualized, highlighting areas with high seismic activity and statistical indices of the dataset were computed providing a comprehensive summary of the features. Then, box plots were created to visualize the distribution of earthquake magnitudes at different depths, grouped by the type of seismic event. This plot provided a clear picture of the interquartile range of earthquake magnitudes for different types of seismic events at various depths.
In the next step, we focused on large (magnitude 7+) earthquakes that occurred between 1906 and 2006. A Poisson model was fitted to the data due to its suitability for modeling count data, its assumption of event independence and its simplicity. The model fitting process involved tuning the model parameters to maximize the log-likelihood of the observed data given the model and the best model was selected based on this score.
Then, the project successfully applied additional statistical analysis and probability models to earthquake prediction. The dataset was first explored to understand the available parameters and their characteristics. A density mapbox plot of earthquake locations was created, which revealed Japan as a region with high seismic activity. This led to a more detailed investigation of earthquakes in Japan. The Epidemic Type Aftershock Sequence (ETAS) model was utilized to analyze the sequence of earthquakes. This model, which incorporates the Omori-Utsu law and a branching process, provided a measure of the seismicity rate at time ‘t’ by considering the effects of all prior earthquakes and the background seismicity rate. The model parameters were determined using Maximum Likelihood Estimation (MLE).
The Python implementation involved filtering the data for earthquakes that occurred in Japan and determining the ETAS parameters. The seismicity rate and transformed time were plotted, which helped to understand how well the ETAS model fits the earthquake data. The plots showed a good fit, as the seismicity rate plot closely matched the distribution of earthquakes over time and the transformed time plot closely followed the diagonal line.
For further investigation, a RandomForest model was also fitted to the data. The number of estimators was adjusted to get the best fit but the R2 score of approximately 0.397 suggested that the RandomForest model might not be a great fit to the data, unlike the Poisson model previously fitted. This highlights the importance of model selection in statistical analysis.
To put it in a nutshell, the analysis has demonstrated the potential of statistical and probabilistic modeling in understanding and predicting earthquake occurrences using rather basic statistical methods alongside Python visualization.

