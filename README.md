# Predictive Maintenance - Predicting the operational statuses of water assets 

### Background

This series of notebook will focus on a  predictive maintenance problem. The business challenge is brought by www.drivendata.org and the goal is to  predict the operation state (functional, repair needed, complete failure) of water pumps in Tanzania. The water pump data is provided by taarifa.org an organization that provides people the ability to report water and sanitation problems in Africa. Here is a dashboard of the status of water points in Tanzania: http://dashboard.taarifa.org/#/dashboard.</br>

![Water Pump](http://drivendata.materials.s3.amazonaws.com/pumps/pumping.jpg) 

source: Pump image courtesy of flickr user christophercjensen


### Business Objectives

Using data from Taarifa and the Tanzanian Ministry of Water, we will be trying to predict which pumps are functional, which need some repairs, and which don't work at all. Predictions will be based on on a number of variables such as:  the type of water pump, when it was installed, how it is managed , where it is located etc. A good understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania</br>

## Link to Analysis and Python Code (Jupyter Notebook)

The code of the project is split into four notebooks: EDA > Data Clean-up & Preprocessing  > Model Fine-tuning | Modelling & Predictions

* ![Exploratory data analysis notebook](https://github.com/ChristopherCochet/Predictive-Maintenance/blob/master/Pump%20it%20Up%20-%20EDA.ipynb)

![Data Cleaning & Feature Engineering & Preprocessing](https://github.com/ChristopherCochet/Predictive-Maintenance/blob/master/Pump%20it%20Up%20-%20EDA.ipynb)

* ![Model Tuning notebook](https://github.com/ChristopherCochet/Predictive-Maintenance/blob/master/Pump%20it%20Up%20-%20Optimize%20Model%20Parameters.ipynb)

* ![Data clean-up, feature engineering and modelling](https://github.com/ChristopherCochet/Predictive-Maintenance/blob/master/Pump%20it%20Up%20-%20Data%20Clean-up.ipynb)


### Custom Code Dependencies

* water_asset_data: defines the Water_Asset_Data class which is used to manage the train and test set panda dataframes and the cleaning and pre-processing the features 
* gis_map_viz.py: defines the visualization GIS_Map_Viz helper class to display the location of water pumps in Tanzania (uses the basemap python package)
* wp_util.py : utility file which defines helper function such as the Cramer's V statistic method for categorical feature correlation analysis

### Python Library Prerequisites

```
Python 2.5 and above
Numpy
Pandas
Matplotlib
Scipy
Scikit-learn
Mxnet
Keras
```

Package             Version  
------------------- ---------
backcall            0.1.0    
basemap             1.1.0    
bleach              2.1.3    
certifi             2018.4.16
chardet             3.0.4    
colorama            0.3.9    
cycler              0.10.0   
decorator           4.3.0    
entrypoints         0.2.3    
graphviz            0.8.4    
h5py                2.8.0    
html5lib            1.0.1    
idna                2.7      
ipykernel           4.8.2    
ipython             6.4.0    
ipython-genutils    0.2.0    
jedi                0.12.0   
Jinja2              2.10     
jsonschema          2.6.0    
jupyter-client      5.2.3    
jupyter-core        4.4.0    
Keras               2.2.0    
Keras-Applications  1.0.2    
keras-mxnet         2.2.0    
Keras-Preprocessing 1.0.1    
kiwisolver          1.0.1    
MarkupSafe          1.0      
matplotlib          2.2.2    
mistune             0.8.3    
mkl-fft             1.0.0    
mkl-random          1.0.1    
mxnet-cu80          1.2.0    
mxnet-cu90          1.2.0    
nbconvert           5.3.1    
nbformat            4.4.0    
notebook            5.5.0    
numpy               1.14.5   
olefile             0.45.1   
pandas              0.23.1   
pandocfilters       1.4.2    
parso               0.2.1    
patsy               0.5.0    
pickleshare         0.7.4    
Pillow              5.1.0    
pip                 10.0.1   
prompt-toolkit      1.0.15   
Pygments            2.2.0    
pyparsing           2.2.0    
pyproj              1.9.5.1  
pyshp               1.2.12   
python-dateutil     2.7.3    
pytz                2018.5   
pywinpty            0.5.4    
PyYAML              3.13     
pyzmq               17.0.0   
requests            2.19.1   
scikit-learn        0.19.1   
scipy               1.1.0    
seaborn             0.8.1    
Send2Trash          1.5.0    
setuptools          39.2.0   
simplegeneric       0.8.1    
six                 1.11.0   
statsmodels         0.9.0    
terminado           0.8.1    
testpath            0.3.1    
tornado             5.0.2    
traitlets           4.3.2    
urllib3             1.23     
wcwidth             0.1.7    
webencodings        0.5.1    
wheel               0.31.1   
wincertstore        0.2


## Author

* **Chris Cochet** - *All work* - [Chris Cochet](https://github.com/ChristopherCochet)