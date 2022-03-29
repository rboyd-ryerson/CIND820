#!/usr/bin/env python
# coding: utf-8

# ## CIND820 Project Course Code
# #### Ryan Boyd - 501072988
# ### Assignment 3
# ### Initial Results and Code

# In[1]:


#import libraries
import laspy as lp, sklearn as skl, numpy as np, matplotlib as mp, pandas as pd


# In[2]:


from sklearn import cluster
from sklearn import preprocessing as prep
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, calinski_harabasz_score
from sklearn.cluster import OPTICS


# In[3]:


from scipy.spatial import ConvexHull, Voronoi


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import PIL
from PIL import ImageStat as istat
from PIL import ImageOps


# #### Data is loaded and prepared

# In[6]:


#original dataset https://nrs.objectstore.gov.bc.ca/gdwuts/092/092g/2016/dsm/bc_092g025_3_4_2_xyes_8_utm10_20170601_dsm.laz
#renamed here for clarity
path_to_data = "F:/Data/Lidar/dtvan/dtvan.laz"
with lp.open(path_to_data) as las_file:
    las_data = las_file.read()


# In[7]:


# data loaded into a dataframe
df = pd.DataFrame({"X":las_data.x,"Y":las_data.y,"Z":las_data.z,"Intensity":las_data.intensity,"return_num":las_data.return_number,"totalreturns":las_data.num_returns,"classification":las_data.classification})


# In[8]:


#full dataset displayed on a scatter plot
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df['X'],df['Y'],zorder=1,alpha=0.25,c='black',s=0.001)


# In[9]:


print("Total points:" + str(las_data.header.point_count))


# In[10]:


#print("Classes: " + str(set(list(df['classification']))))


# In[11]:


#data summary
df.describe()


# #### The full original dataset is too large to work with so it is clipped to a smaller study area. The dataframe is queried by the study area longitude and latitude boundary. The pre-classified ground points, class 2, are removed (since we are not concerned with ground right now), class 1 are the unclassified points and we only want to work with these.

# In[12]:


# Define the area of interest, these values in meters are in the "NAD83 UTM 10" coordinate system of the provided dataset
# These are the upper and lower limits in meters to be used, these can be found using google maps or other free sites/software
# These were selected somewhat arbitrarily
aoi_extent = {'xmax':492349.0731766,'xmin':492043.6935073,'ymax':5458645.8660691,'ymin':5458340.4864470}


# In[285]:


#query the dataframe to return only the points within the extent above and remove the points defined as ground as well
df_clip = df.query("X>{0}&X<{1}&Y>{2}&Y<{3}&Intensity<200".format(aoi_extent['xmin'],aoi_extent['xmax'],aoi_extent['ymin'],aoi_extent['ymax']))


# In[385]:


df_clip.describe()


# #### Dataset statistics and information - exploratory

# In[386]:


#renaming the data frame for clarity
data = df_clip


# In[387]:


mp.pyplot.hist(data['totalreturns'])


# In[388]:


mp.pyplot.hist(data['Y'])


# In[389]:


mp.pyplot.hist(data['Z'])


# In[390]:


mp.pyplot.hist(data['Intensity'])
#was very heavy on the low value and light on the high so queried to only 300 or less


# In[391]:


i_cutoff = 300


# In[392]:


len(data[data['Intensity']<i_cutoff])


# In[393]:


len(data[data['Intensity']>i_cutoff])


# In[394]:


mp.pyplot.hist(data[data['Intensity']<i_cutoff])


# In[395]:


len(data[data['Intensity']>i_cutoff])/len(data[data['Intensity']<i_cutoff])


# In[396]:


#summarize the normalized data
data.describe()


# In[397]:


#study area points displayed on a scatter plot
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(data['X'],data['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[398]:


#the height is used value to visualize in 3d, since the values are in meters for all 3 axes, it plots nicely as is
fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(data['X'],data['Y'],data['Z'],c='black',s=0.01,alpha=0.5)

#from matplotlib import cm
#ax.plot_surface(df_clip['X'],df_clip['Y'],df_clip['Z'],cmap=cm.coolwarm,linewidth=0,antialiased=False)


# # DEM Height Values

# In[399]:


# The Z values are an absolute elevation which is not as useful as height relative to the ground
# I need to create a ground image used to subtract from the elevation(Z) in order to get the height of the points relative to the ground


# The DEM was provided with the lidar data, I can clip this and extract the elevation of the ground for the entire area. Where the is something on the ground such as a building, the value is estimated using the nearest ground points available. I can then subtract the laser return value by the ground value to get the relative height of the object instead of the absolute height to sea level. This gives a more accurate height value to use with the algorithms.

# In[400]:


dem_path = "F:/Data/Lidar/images/BCVAN_DEM1m_Clip.tif"
img0 = PIL.Image.open(dem_path)


# In[401]:


dem_array = np.asarray(img0)


# In[415]:


dem_img = img0.convert("F")


# In[416]:


#dem_img = img0.convert("RGB")


# In[417]:


np.asarray(dem_img)


# In[418]:


x_dem_img = (data['X'] - min(data['X']))/(max(data['X']-min(data['X'])))*306
y_dem_img = (data['Y'] - min(data['Y']))/(max(data['Y']-min(data['Y'])))*306


# In[419]:


x_dem_img


# In[420]:


coord_array_dem = np.array(pd.DataFrame({"X":x_dem_img,"Y":y_dem_img}))


# In[421]:


coord_array_dem


# In[422]:


dem_value = []
for coord in coord_array_dem:
    val = dem_img.getpixel((coord[0],coord[1]))
    dem_value.append(val)


# In[423]:


len(dem_value)


# In[424]:


data['dem_value'] = dem_value


# In[425]:


data


# In[426]:


data['height'] = data['Z'] - data['dem_value']


# In[427]:


data['height'].describe()


# In[428]:


data


# In[429]:


df_unclassified = data.query("classification==1")
df_ground = data.query("classification==2")


# In[430]:


# Plotting the pre classified/labelled ground points for reference
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df_ground['X'],df_ground['Y'],c='black',s=0.01,alpha=0.5)


# In[431]:


# Plotting the pre classified/labelled unclassified points for reference
#it appears that alot of ground points are still labelled as unclassified.
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df_unclassified['X'],df_unclassified['Y'],c='black',s=0.01,alpha=0.5)


# # Normalization

# #### Data normalized and preprocessed for analysis

# In[432]:


x_normal = (data['X'] - min(data['X']))/(max(data['X']-min(data['X'])))


# In[433]:


y_normal = (data['Y'] - min(data['Y']))/(max(data['Y']-min(data['Y'])))


# In[434]:


z_normal = (data['Z'] - min(data['Z']))/(max(data['Z']-min(data['Z'])))


# In[435]:


height_normal = (data['height'] - min(data['height']))/(max(data['height']-min(data['height'])))


# In[436]:


i_normal = (data['Intensity'] - min(data['Intensity']))/(max(data['Intensity']-min(data['Intensity'])))


# In[437]:


# new dataframe containing all the normalized values is created
df_normal = pd.DataFrame({'X':x_normal,'Y':y_normal,'Z':z_normal,'height':height_normal,'Intensity':i_normal,'return_num':df_clip['return_num'],'totalreturns':df_clip['totalreturns'],'classification':df_clip['classification']})


# In[438]:


df_normal


# In[439]:


# Plotting normalized looks the same but with the new scale
fig,ax = plt.subplots(figsize = (10,10))
ax.scatter(df_normal['X'],df_normal['Y'],c='black',s=0.01,alpha=0.5)


# In[440]:


df_normal.dtypes


# # Supervised Classification
# ## Ground

# In[441]:


# Classify the ground for supervised classifier using the provided ground points as labels:


# In[442]:


df_normal


# In[443]:


train,test = train_test_split(df_normal)


# In[444]:


train_features = pd.DataFrame({"Intensity":train['Intensity'],"return_num":train['return_num'],"totalreturns":train['totalreturns'],"height":train['height']})


# In[445]:


train_labels = np.ravel(pd.DataFrame({"classification":train['classification']}))


# In[446]:


test_features = pd.DataFrame({"Intensity":test['Intensity'],"return_num":test['return_num'],"totalreturns":test['totalreturns'],"height":test['height']})


# In[447]:


test_labels = np.ravel(pd.DataFrame({"classification":test['classification']}))


# In[448]:


#creates the model
model = RandomForestClassifier(max_depth=5,random_state=0,n_estimators=50,criterion="entropy",verbose=0,class_weight="balanced")


# In[449]:


# trains the model - fit the train data to the model
model_fit = model.fit(train_features,train_labels)


# In[450]:


#predict the test data
test_predictions = model_fit.predict(test_features)


# In[451]:


len([i for i in test_predictions if i == 1])


# In[452]:


len([i for i in test_predictions if i != 1])


# In[453]:


model_fit.score(test_features,test_labels)


# In[454]:


confusion_matrix(test_labels,test_predictions)


# In[455]:


table = pd.DataFrame({"Intensity":df_normal['Intensity'],"return_num":df_normal['return_num'],"totalreturns":df_normal['totalreturns'],"H":df_normal['height']})


# In[456]:


table_labels = df_normal['classification']


# In[457]:


table_predictions = model_fit.predict(table)


# In[458]:


len([i for i in table_predictions if i == 1])


# In[459]:


len([i for i in table_predictions if i != 1])


# In[460]:


model_fit.score(table,table_labels)


# In[461]:


confusion_matrix(table_labels,table_predictions)


# In[462]:


df_normal['prediction'] = table_predictions.tolist()


# In[463]:


df_normal


# In[464]:


df_normal.query("classification != prediction")


# In[465]:


predicted_ground = df_normal.query("prediction == 2")


# In[466]:


predicted_ground


# In[467]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(predicted_ground['X'],predicted_ground['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[468]:


last_ground = predicted_ground.query("return_num==totalreturns")


# In[469]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(last_ground['X'],last_ground['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[470]:


last_ground


# In[471]:


predicted_non_ground = df_normal.query("prediction == 1")


# In[472]:


fig,ax = plt.subplots(figsize = (10,10))
ax.scatter(predicted_non_ground['X'],predicted_non_ground['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[473]:


predicted_non_ground


# In[474]:


data = predicted_non_ground


# In[475]:


data


# ## Add Imagery Data
# #### 2015 Imagery data was obtained from the City of Vancouver to extract the RGB values
# #### The image was clipped using external software (QGIS, open-source mapping program) to the same area of interest as above
# #### The selected image size is 4084x4084, the lidar data is normalized by 4084 to extract the nearest pixel value(r,g,b) from the image for each point

# In[476]:


image_path = "F:/Data/Lidar/images/BCVANC15_P9_aoiclip.tif"


# In[477]:


img = PIL.Image.open(image_path)


# In[478]:


rgb_img = img.convert("RGB")


# In[479]:


rgb_img


# In[480]:


#this can be used to crop the imagery if we knew the exact coordinates, but I used QGIS to clip the imagery instead
#left,top,right,bottom = 0,0,4084,4084
#rgb_img = img.crop((left,top,right,bottom))


# In[481]:


#import math
#math.sqrt(istat.Stat(rgb_img).count[0])


# In[482]:


#this size aligns the pixels to the lidar points to extract the rgb values for each point
img.size


# In[483]:


#The image origin (top left) is different than the coordinate system of the lidar so the image needs to be flipped for the calculation to align them
rgb_img_flip = PIL.ImageOps.flip(rgb_img)


# In[484]:


#rgb_img.getpixel((2070,2070))


# In[485]:


# rescales the point values to line up with the pixels in the imagery - same idea as normalization
# this is basically reprojecting the coordinates of the lidar points to the coordinates of the image
x_img = (data['X'] - min(data['X']))/(max(data['X']-min(data['X'])))*4083
y_img = (data['Y'] - min(data['Y']))/(max(data['Y']-min(data['Y'])))*4083


# In[486]:


y_img


# In[487]:


coord_array = np.array(pd.DataFrame({"X":x_img,"Y":y_img}))


# In[488]:


coord_array
# locations on the image to read the rgb values


# #### The nearest R,G,B pixel value from the image is extracted for each lidar point and the results are saved as a field in the data frame

# In[489]:


rgb_data = []
rgb_data_r = []
rgb_data_g = []
rgb_data_b = []
for coord in coord_array:
    rgb=rgb_img_flip.getpixel((coord[0],coord[1]))
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    rgb_data.append(rgb)
    rgb_data_r.append(r)
    rgb_data_g.append(g)
    rgb_data_b.append(b)
data['rgb'] = rgb_data
data['r'] = rgb_data_r
data['g'] = rgb_data_g
data['b'] = rgb_data_b


# In[490]:


data


# ## Vegetation - abandoned

# In[491]:


#trying to extract the vegetation using combination of fields for supervised classifier, to varied success
#vegetation = df_all.query("return_num < totalreturns & totalreturns > 1")
#fig,ax = plt.subplots(figsize = (15,15))
#ax.scatter(vegetation['X'],vegetation['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[492]:


vegetation = data.query("return_num < totalreturns & totalreturns > 1")
#vegetation = data.query("r<0.9 & g>0.1")


# In[493]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(vegetation['X'],vegetation['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[494]:


data_normal = data


# #### The R,G,B values are normalized like the rest:

# In[495]:


r_normal = (data['r'] - min(data['r']))/(max(data['r']-min(data['r'])))
g_normal = (data['g'] - min(data['g']))/(max(data['g']-min(data['g'])))
b_normal = (data['b'] - min(data['b']))/(max(data['b']-min(data['b'])))
data_normal['r'] = r_normal
data_normal['g'] = g_normal
data_normal['b'] = b_normal


# In[496]:


data_normal


# #### Additonal testing and improvements

# In[497]:


data = data_normal


# In[498]:


data


# #### Initial Classification (unsupervised) using kmeans clustering
# #### Attempt to classify points into undetermined classes based on the data
# ##### Variables: Height, Intensity, R, G, B

# In[499]:


#features = pd.DataFrame({"R":data['r'],"G":data['g'],"B":data['b'],"H":data['height'],"I":data['Intensity']})
#features = pd.DataFrame({"R":data['r'],"G":data['g'],"B":data['b']})
features = pd.DataFrame({"H":data['height'],"I":data['Intensity']})


# In[500]:


features


# In[501]:


X1 = np.array(features)


# In[502]:


X1


# In[503]:


# initialize model
kmeancluster = cluster.KMeans(3,init='random',n_init=10)


# In[504]:


# fit to data
k_clusters = kmeancluster.fit(X1)


# In[505]:


len(k_clusters.labels_)


# In[506]:


print("Number of clusters:" + str(len(np.unique(k_clusters.labels_))))
print("Points clustered: " + str(len([i for i in k_clusters.labels_ if i != -1])))


# In[507]:


calinski_harabasz_score(X1,k_clusters.labels_)


# In[547]:


silhouette_score(X1,k_clusters.labels_,sample_size=10000)


# In[508]:


# add results to data frame
data['k_cluster'] = k_clusters.labels_


# In[509]:


#rename for clarity again
results = data


# In[510]:


# visualize the classes
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(results['X'],results['Y'],c=results['k_cluster'],s=0.1,alpha=0.5)


# In[511]:


fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(results['X'],results['Y'],results['Z'],c=results['k_cluster'],s=0.01,alpha=1)


# In[512]:


#try to remove shadow
colour_manip = pd.DataFrame()


# In[513]:


results


# #### Spatial/Distance Clustering

# #### This section will attempt to use the previous classificiation label to cluster the points into local distinct objects

# In[573]:


df2 = pd.DataFrame({"X":data['X'],"Y":data['Y'],"K":data['k_cluster']})
#df2 = pd.DataFrame({"X":data['X'],"Y":data['Y'],"Z":data['height'],"r":data['r'],"g":data['g'],"b":data['b'],"I":data['Intensity']})
#df2 = pd.DataFrame({"X":data['X'],"Y":data['Y'],"H":data['height'],"I":data['Intensity']})
#df2 = pd.DataFrame({"H":data['height'],"I":data['Intensity']})


# In[574]:


X2 = np.array(df2)


# In[575]:


X2


# In[576]:


# to determine the value for the 'eps' parameter, the value of the inflection point on this distance graph is used
# this is the average distance between the points
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X2)
distances,indices = nbrs.kneighbors(X2)
distances = np.sort(distances,axis=0)
distances = distances[:,1]
plt.plot(distances)


# In[577]:


#create the model
dbscan = cluster.DBSCAN(eps=0.005,min_samples=50,algorithm="auto",leaf_size=30,n_jobs=-1)


# In[578]:


#db_clusters = OPTICS(min_samples=50).fit(X2)


# In[579]:


# fit the data to the model to get the cluster id's
db_clusters = dbscan.fit(X2)


# In[580]:


cluster_count = str(len(np.unique(db_clusters.labels_)))
points_clustered = str(len([i for i in db_clusters.labels_ if i != -1]))
print("Number of clusters:" + cluster_count)
print("Points clustered: " + points_clustered)


# In[581]:


labels = db_clusters.labels_
calinski_harabasz_score(X2,labels)


# In[582]:


silhouette_score(X2,db_clusters.labels_,sample_size=20000)


# In[583]:


#add cluster results to dataframe
results['db_cluster'] = list(db_clusters.labels_)


# In[584]:


#filter out results not clustered
results2 = results.query("db_cluster!=-1")


# In[585]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(results2['X'],results2['Y'],c=results2['db_cluster'],s=0.1,alpha=0.5)


# In[586]:


fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(results2['X'],results2['Y'],results2['Z'],c=results2['db_cluster'],s=0.01,alpha=1)


# # Filtering Results

# In[587]:


#looking at individual clusters which each should be equivalent to a unique/distinct "object"
result = results2.query("db_cluster>=0&db_cluster<={0}".format(cluster_count))
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(predicted_ground['X'],predicted_ground['Y'],c='grey',s=10,alpha=1)
ax.scatter(result['X'],result['Y'],c=result['db_cluster'],s=5,alpha=1)


# In[588]:


fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
#ax.scatter3D(predicted_ground['X'],predicted_ground['Y'],predicted_ground['Z'],c='grey',s=10,alpha=1)
ax.scatter3D(results2['X'],results2['Y'],results2['Z'],c=results2['db_cluster'],s=5,alpha=1)


# #### Analysis of results

# -supervised ground (from preclassified points)
#     -random forest
#     -
# -supervised vegetation (from known points)
# -grid search for parameters
# -randomized forest/svm
# -eigenvalues
# -adjust or convert rgb values of the shadow class

# In[589]:


#imagery again for reference
rgb_img


# In[613]:


result = results2.query("db_cluster>=0 & db_cluster<={0}".format(cluster_count))
fig,ax = plt.subplots(figsize = (25,25))
ax.imshow(rgb_img, extent=[0, 1, 0, 1])
ax.scatter(result['X'],result['Y'],c=result['db_cluster'],s=0.5,alpha=1)
#ax.scatter(predicted_ground['X'],predicted_ground['Y'],c='black',s=10,alpha=1)


# In[591]:


point_counts = []
small,medium,large = [],[],[]
for i in result['db_cluster'].unique():
    cluster_num = i
    point_count = len(result.query("db_cluster=={0}".format(i)))
    point_counts.append(point_count)
    print(str(cluster_num)+": "+str(point_count))
    if point_count < 500:
        small.append(cluster_num)
    elif point_count >= 500 and point_count < 2500:
        medium.append(cluster_num)
    elif point_count >= 2500:
        large.append(cluster_num)


# In[ ]:





# In[592]:


print(" small: " + str(len(small)) + " medium: " + str(len(medium)) + " large: " + str(len(large)))


# In[593]:


alluniqueclusters = result['db_cluster'].unique()


# In[594]:


# small clusters
a,b = plt.subplots(figsize = (25,25))
b.imshow(rgb_img, extent=[0, 1, 0, 1])
for i in small:
    clust = result.query("db_cluster=={0}".format(i))
    points = []
    for j in range(len(clust)):
        points.append([list(clust['X'])[j],list(clust['Y'])[j]])
    if len(points)>3:
        point_array = np.array(points)
        hull = ConvexHull(point_array)
        b.plot(point_array[:,0],point_array[:,1],'o')
    for s in hull.simplices:
        b.plot(point_array[s,0],point_array[s,1],'k-')


# In[595]:


#medium clusters
a,b = plt.subplots(figsize = (25,25))
b.imshow(rgb_img, extent=[0, 1, 0, 1])
for i in medium:
    clust = result.query("db_cluster=={0}".format(i))
    points = []
    for j in range(len(clust)):
        points.append([list(clust['X'])[j],list(clust['Y'])[j]])
    if len(points)>3:
        point_array = np.array(points)
        hull = ConvexHull(point_array)
        b.plot(point_array[:,0],point_array[:,1],'o')
    for s in hull.simplices:
        b.plot(point_array[s,0],point_array[s,1],'k-')


# In[596]:


#large clusters
a,b = plt.subplots(figsize = (25,25))
b.imshow(rgb_img, extent=[0, 1, 0, 1])
for i in large:
    clust = result.query("db_cluster=={0}".format(i))
    points = []
    for j in range(len(clust)):
        points.append([list(clust['X'])[j],list(clust['Y'])[j]])
    if len(points)>3:
        point_array = np.array(points)
        hull = ConvexHull(point_array)
        b.plot(point_array[:,0],point_array[:,1],'o')
    for s in hull.simplices:
        b.plot(point_array[s,0],point_array[s,1],'k-')


# In[153]:


a,b = plt.subplots(figsize = (25,25))
b.imshow(rgb_img, extent=[0, 1, 0, 1])
for i in alluniqueclusters:
    clust = result.query("db_cluster=={0}".format(i))
    points = []
    for j in range(len(clust)):
        points.append([list(clust['X'])[j],list(clust['Y'])[j]])
    if len(points)>3:
        point_array = np.array(points)
        hull = ConvexHull(point_array)
        b.plot(point_array[:,0],point_array[:,1],'o')
    for s in hull.simplices:
        b.plot(point_array[s,0],point_array[s,1],'k-')


# In[154]:


b.scatter(predicted_ground['X'],predicted_ground['Y'],c='black',s=15,alpha=1)


# In[155]:


a


# In[156]:


new_classes = ["building", "vegetation"]


# # Next Steps

# In[157]:


# manually label each feature


# In[158]:


# retrain the previous classifier to apply those labels

