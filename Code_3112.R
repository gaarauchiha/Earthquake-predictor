suppressWarnings(suppressMessages(library(forecast)))
suppressWarnings(suppressMessages(library(data.table)))
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(corrplot)))
suppressWarnings(suppressMessages(library(astsa)))
suppressWarnings(suppressMessages(library(maps)))
suppressWarnings(suppressMessages(library(plyr)))
suppressWarnings(suppressMessages(library(fpp)))
suppressWarnings(suppressMessages(library(lubridate)))



database <- fread("EarthQuake.csv",stringsAsFactors = T) #Read the data
database$Date <- as.Date(database$Date, format="%d/%m/%Y") #Sets format to dates
database <- database[Type=="Earthquake"] #We're gonna use only the earthquakes
database <- database[,c("ID","Date","Time","Latitude","Longitude","Magnitude")] #Lets clear some data
database <- database[complete.cases(database[,2]),] #Drop the cases with NAs on the Dates
database$Year <- format(as.Date(database$Date, format="%d/%m/%Y"),"%Y") #We'll need the years more ahead
summary(database)



map <- ggplot(database) + borders("world", colour="black", fill="gray50")  
print(map + geom_point(aes(x=database$Longitude, y=database$Latitude,color=Magnitude),shape=18) +
        scale_color_gradient(low="blue", high="red") +
        theme(legend.position = "top")+
        ggtitle("Earthquakes by Magnitude"))


ggplot(database,aes(Magnitude))+
  geom_area(aes(y = ..count..,fill="blue"), stat = "bin")+
  labs(title="Earthquakes",caption="jhervas") + 
  guides(fill=FALSE)


magnitudes_over_years <- ddply(database, .(Year), summarize,  Mean_Magnitude=mean(Magnitude))

Magnitudes <- ts(magnitudes_over_years[2],
                 start=1965, #min(database$Date, na.rm=TRUE)
                 end=2016, #max(database$Date, na.rm=TRUE),
                 frequency =1)
plot(Magnitudes)




Earthquakes <- ts(unname(table(database$Year)),
                  start=1965, #min(database$Date, na.rm=TRUE)
                  end=2023, #max(database$Date, na.rm=TRUE),
                  frequency =1)
plot(Earthquakes)



diff_Earthquakes <- diff(Earthquakes)
diff_Magnitudes <- diff(Magnitudes)
par(mfrow=c(2,1))
plot(diff_Earthquakes)
plot(diff_Magnitudes)


Box.test(diff_Earthquakes, lag=20, type="Ljung-Box")
Box.test(diff_Magnitudes, lag=20, type="Ljung-Box")

adf.test(diff_Earthquakes, alternative ="stationary")
adf.test(diff_Magnitudes, alternative ="stationary")

kpss.test(diff_Earthquakes)
kpss.test(diff_Magnitudes)



plot(acf2(diff_Earthquakes))



sarima(Earthquakes, 1, 1, 1)





sarima(Earthquakes, 2, 2, 2)





plot(acf2(diff_Magnitudes))




par(mfrow=c(2,1))
sarima.for(Earthquakes, n.ahead=5, 1, 1, 1)
sarima.for(Magnitudes,n.ahead=5, 1, 1, 1)











































