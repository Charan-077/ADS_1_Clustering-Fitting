# Importing Libraries
import pandas as pnds_
import matplotlib.pyplot as mat_plt
from matplotlib.figure import Figure
import numpy as nmpi
import matplotlib as mbs
from scipy.optimize import curve_fit
import csv
from sklearn.cluster import KMeans


def getting_data(file):
    """This function takes in a csv file in the Worldbank format and returns a transposed version as well as the original
    version in the dataframe format of pandas.
    
    Parameters
    ----------
    csv_file : str
        The filename of the csv file.

    Returns
    -------
    final : pandas dataframe
        A dataframe containing the original csv data.
    transposed_csv : pandas dataframe
        A dataframe containing the transposed csv data.
    """
    #empty list created to store the data
    lst = []
    #opening the file in read mode
    with open (file, 'r') as f_handle:
        #reading it using csv.reader of python
        reading = csv.reader(f_handle, delimiter=",")
        #appending the data values in the list
        for k in reading:
            lst.append(k)
    #converting list into a dataframe
    data = pnds_.DataFrame(lst)
    #transposing the dataframe
    data_t = data.T
    #returns original dataframe and transposed dataframe
    return data, data_t


main, transposed = getting_data("API_19_DS2_en_csv_v2_4756035.csv")


main.head(2)


main = main[4:]


main.columns = main.iloc[0]


main.drop(4, inplace=True)


main.rename(columns={'Country Name':'Country_Name','Country Code':'Country_Code','Indicator Name':'Indicator_Name','Indicator Code':'Indicator_Code'},inplace=True)


main


transposed


main=main.replace(nmpi.nan,0)
main.head()


main.isnull().sum()


main.columns


main.dtypes


main.replace("", 0, inplace=True)


main[main.columns[4:68]] = main[main.columns[4:68]].astype(float)


main.info()

main.describe()


main.shape


Indicator_Name=main.groupby(['Country_Name','Country_Code','Indicator_Name', 'Indicator_Code','1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']).size().reset_index()
Indicator_Name.head(2)


Indicator_Name.set_index("Indicator_Name",inplace = True)
forest = Indicator_Name.loc["Forest area (% of land area)"]
forest.head(2)


forest = forest.drop(columns= list(forest.columns)[1:35],axis =1)


forest


forest1 = forest[forest.columns[2:65]]



forest1


yax=['1993', '1996', '1998', '2000', '2005', '2010', '2016']
forestSample =forest[::37]
forestSample.plot(x= "Country_Name", y = yax,kind = "bar",figsize = (13,7),logy = True,color=["#ea5545", "#f46a9b", "#ef9b20", "#edbf33", "#ede15b", "#bdcf32", "#87bc45", "#27aeef", "#b33dc6"])
mat_plt.title(" Annual Growth in Forest areas (% of land area) (Country Wise)", fontsize='20')


cluster_handle = KMeans(n_clusters=3)


values = cluster_handle.fit_predict(forest1)


centre_values = cluster_handle.cluster_centers_


values


val_data = pnds_.DataFrame(values)


val_data=val_data.rename(columns = {0:"Cluster"})


val_data


index = list(val_data.index.values)
forest["index"]=index
forest.set_index(["index"], drop=True)


val_data["index"]=index
val_data.set_index(["index"], drop=True)


final_forest = pnds_.merge(forest, val_data)


final_forest.drop(columns = ["index"], inplace = True)




final_forest


lst1=[]
lst2=[]
#getting the centroid value of only 1993 and 2020
for i in centre_values:
    for j in range(len(i)):
        x = i[2]
        y = i[27]
    lst1.append(i[2])
    lst2.append(i[27])


centre_values


# selection of the colours for the addition to the graph
colors = ['#fc1313', '#fa15f3', '#9b13ff']
#cluster based mapping of the colors
final_forest['clr'] = final_forest.Cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
#i figure size initiated
fig, ax = mat_plt.subplots(1, figsize=(13,7))
#plotting a scatter plot of data
mat_plt.scatter(final_forest["1993"], final_forest["2020"], c=final_forest.clr, alpha = 0.6, s=40)
#plotting a scatter plot of centroids
mat_plt.scatter(lst1, lst2, marker='o',facecolor=colors,edgecolor="black", s=100)
#getting the legend for data
legend_elements = [mbs.lines.Line2D([0], [0], marker='o', color='w', label='Cluster or C{}'.format(i+1), 
                   markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
#getting the legend for centroids
centroid_legend = [mbs.lines.Line2D([0], [0], marker='o', color='w', label='Centroid of C{}'.format(i+1), 
                   markerfacecolor=mcolor,markeredgecolor="black", markersize=10) for i, mcolor in enumerate(colors)]
#final legend elements
legend_elements.extend(centroid_legend)
#setting the legend
mat_plt.legend(handles=legend_elements, loc='upper right', title="Clusters", fontsize=10, bbox_to_anchor=(1.15,1))
#setting xlabel, ylabel and title
mat_plt.xlabel("1993 data in %", fontsize='18')
mat_plt.ylabel("2020 data in %", fontsize='18')
mat_plt.title("Basic K-means clustering for analysis of forest areas", fontsize='20')


def fit_data(dataframe, col_x, col_y):
    """    
    Parameters
    ----------
    dataframe : pandas dataframe
        The name of the pandas dataframe.
    col_x : str
        The name of the column of dataframe to be taken as x.  
    col_y : str
        The name of the column of dataframe to be taken as y.  

    Returns
    -------
    errors : numpy array
        An array containing the errors or ppot.
    covariance : numpy array
        An array containing the covariances.
    """
    x_data = dataframe[col_x]
    y_data = dataframe[col_y]
    
    def mod_func(x, m, b):
        return m*x+b
        
    #calling curve_fit
    errors,covariance = curve_fit(mod_func, x_data, y_data)
    mat_plt.figure(figsize=(13,7))
    #plotting the data 
    mat_plt.plot(x_data, y_data, "bo", label="clusters", color ="b")
    #plotting the best fitted line
    mat_plt.plot(x_data, mod_func(x_data, *errors), "k-", label="Best Fit")
    mat_plt.xlabel(col_x)
    mat_plt.ylabel(col_y)
    mat_plt.legend(bbox_to_anchor=(1,1))
    mat_plt.title("Plotting best fit for the Annual Growth in Forest areas", fontsize='20')
    return errors, covariance


fit_data(final_forest,"2000","2011")


#Data points 
x_data = final_forest["2000"]
y_data = final_forest["2011"]
#Defining the Logistic Function
def logistic(x,a,b,c):
    return c/(1 + nmpi.exp(-a*(x-b)))

#Fitting the data to the logistic function
params, cov = curve_fit(logistic, x_data, y_data)

#Calculating the lower and upper limits of the confidence range
def err_ranges(x, y, params):
    y_fit = logistic(x, *params)
    y_res = y - y_fit
    sse = nmpi.sum(y_res**2)
    var = sse / (len(x) - len(params))
    
    ci = 1.96 # 95% confidence interval
    err = ci * nmpi.sqrt(nmpi.diag(cov)*var)
    return nmpi.array([y_fit - err, y_fit + err])

#Predicting values in 10 years time
x_pred = 20
y_pred = logistic(x_pred, *params)
mat_plt.figure(figsize=(13,7))

#Plotting the data and the best fitting logistic function
mat_plt.plot(x_data, y_data, 'o', label='Data Points')
x_fit = lst2
y_fit = logistic(x_fit, *params)
mat_plt.plot(x_fit, y_fit, label='Best Fit')

#Plotting the confidence range
err_range = err_ranges(x_fit, y_fit, params)
print(err_range)
#Marking the predicted value
mat_plt.scatter(x_pred, y_pred, color='red', label='Predicted Value')

#Formatting the plot
mat_plt.xlabel("1993")
mat_plt.ylabel('2020')
mat_plt.title('Logistic Function')
mat_plt.legend(bbox_to_anchor=(1,1))
mat_plt.show()



