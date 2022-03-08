## CIND820
#### Ryan Boyd
## This repository contains the project files for the CIND 820 Big Data Project Course

Repo contents:
-Jupyter notebook (.ipynb) containing the current state of the code, with headings and comments where applicable
-HTML version of the report
-Research Literature PDFs for reference
-Test Notebooks: folder containing older versions of code written for testing purposes and fact finding
-Test_Data: folder containing sample datasets, similar in format and content to the data used for the project, in LAS format. The full dataset used in the code was too large to upload to github so this was included instead.
-Project_Documents: Additional project info and notes including the past assignments for reference

## Project Info

Project Goal: Apply unsupervised learning techniques to lidar data points in order to attempt to identify and discriminate objects within the scene.

Data: .LAS raw lidar data point cloud
The full las dataset and imagery used in this project can be retrieved from 
https://governmentofbc.maps.arcgis.com/apps/MapSeries/index.html?appid=d06b37979b0c4709b7fcf2a1ed458e03

BCGS Tile Name: 092g025_3_4_2
BCGS Grid Scale: 1:2500
File Name: bc_092g025_3_4_2_xyes_8_utm10_20170713.laz
Collected: July 2017

The lidar data is provided under the Open Government License - British Columbia:
https://www2.gov.bc.ca/gov/content/data/open-data/open-government-licence-bc
https://www2.gov.bc.ca/gov/content/data/geographic-data-services/lidarbc

The orthophoto imagery was collected in 2015 and can be downloaded here:
https://opendata.vancouver.ca/explore/dataset/orthophoto-imagery-2015/information/

Image Tile Name: BCVANC15_P9
Download Link: https://webtransfer.vancouver.ca/opendata/2015ecw/BCVANC15_P9.zip
Collected: April-July 2015

Imagery data is provided under the Open Government License - Vancouver:
https://opendata.vancouver.ca/pages/licence/

Tools/Software: Python (Jupyter-notebooks, Numpy, Scikit-learn, Pandas, Matplotlib)
All software is open source and free to use under GPLv3 or BSD License
https://www.gnu.org/licenses/gpl-3.0.en.html
https://opensource.org/licenses/BSD-3-Clause

Methods/Techniques: K-Means, DBSCAN Clustering

Tentative Project Stages:

Initial Subject Research and Review
Research Question Formation
Literature Review
Initial Code and Results
	Steps performed in the code:
		- Libraries and data loaded and prepared
		- Data exploratory analysis
		- Orthoimagery data extraction		
		- Data preprocessing and normalization
		- Classification
		- Clustering
		- Results Analysis
		- Adjustment and testing
		- Iterate above steps
Final Results


