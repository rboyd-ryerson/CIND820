#!/usr/bin/env python
# coding: utf-8

# ## CIND820 Project Course Code
# #### Ryan Boyd - 501072988
# ### Assignment 4
# ### Final Results

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


# In[6]:


import datetime


# #### Data is loaded and prepared

# In[7]:


#original dataset https://nrs.objectstore.gov.bc.ca/gdwuts/092/092g/2016/dsm/bc_092g025_3_4_2_xyes_8_utm10_20170601_dsm.laz
#renamed here for clarity
path_to_data = "F:/Data/Lidar/dtvan/dtvan.laz"
with lp.open(path_to_data) as las_file:
    las_data = las_file.read()


# In[8]:


# data loaded into a dataframe
df = pd.DataFrame({"X":las_data.x,"Y":las_data.y,"Z":las_data.z,"Intensity":las_data.intensity,"return_number":las_data.return_number,"total_returns":las_data.num_returns,"classification":las_data.classification})


# In[9]:


#full dataset displayed on a scatter plot
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df['X'],df['Y'],zorder=1,alpha=0.25,c='black',s=0.001)


# In[10]:


print("Total points:" + str(las_data.header.point_count))


# In[11]:


#print("Classes: " + str(set(list(df['classification']))))


# In[12]:


#data summary
df.describe()


# #### The full original dataset is too large to work with so it is clipped to a smaller study area. The dataframe is queried by the study area longitude and latitude boundary. The pre-classified ground points, class 2, are removed (since we are not concerned with ground right now), class 1 are the unclassified points and we only want to work with these.

# In[13]:


# Define the area of interest, these values in meters are in the "NAD83 UTM 10" coordinate system of the provided dataset
# These are the upper and lower limits in meters to be used, these can be found using google maps or other free sites/software
# These were selected somewhat arbitrarily
aoi_extent = {'xmax':492349.0731766,'xmin':492043.6935073,'ymax':5458645.8660691,'ymin':5458340.4864470}


# In[14]:


#query the dataframe to return only the points within the extent above and remove the points defined as ground as well
df_clip = df.query("X>{0}&X<{1}&Y>{2}&Y<{3}".format(aoi_extent['xmin'],aoi_extent['xmax'],aoi_extent['ymin'],aoi_extent['ymax']))


# #### Dataset statistics and information - exploratory

# In[15]:


df_clip.describe()


# In[16]:


#renaming the data frame for clarity
data = df_clip


# In[17]:


mp.pyplot.hist(data['return_number'])


# In[18]:


mp.pyplot.hist(data['total_returns'])


# In[19]:


#histogram of returns as a ratio
mp.pyplot.hist(data['return_number']/data['total_returns'])


# In[20]:


mp.pyplot.hist(data['X'])


# In[21]:


mp.pyplot.hist(data['Y'])


# In[22]:


mp.pyplot.hist(data['Z'])


# In[23]:


mp.pyplot.hist(data['Intensity'])
#very heavy on the low value and light on the high so adjusted to only 200 or less


# In[24]:


point_count = len(data)
print(point_count)


# In[25]:


i_cutoff = 200


# In[26]:


points_below_cutoff = len(data[data['Intensity']<i_cutoff])
print(points_below_cutoff)
print(points_below_cutoff/point_count)


# In[27]:


points_above_cutoff = len(data[data['Intensity']>i_cutoff])
print(points_above_cutoff)
print(points_above_cutoff/point_count)


# In[28]:


#create a list of the new max value to replace all the value above the cutoff value
vals = []
for i in range(points_above_cutoff):
    vals.append(i_cutoff)


# In[29]:


len(vals)


# In[30]:


len(data.query("Intensity>200"))


# In[31]:


data['Intensity'].mask(data['Intensity'] > 200,200,inplace=True)


# In[32]:


len(data.query("Intensity>200"))


# In[33]:


mp.pyplot.hist(data['Intensity'])


# In[34]:


#summarize the normalized data
data.describe()


# In[35]:


#study area points displayed on a scatter plot
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(data['X'],data['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[36]:


#the height is used value to visualize in 3d, since the values are in meters for all 3 axes, it plots nicely as is
fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(data['X'],data['Y'],data['Z'],c='black',s=0.01,alpha=0.5)

#from matplotlib import cm
#ax.plot_surface(df_clip['X'],df_clip['Y'],df_clip['Z'],cmap=cm.coolwarm,linewidth=0,antialiased=False)


# # DEM Height Values

# In[37]:


# The Z values are an absolute elevation which is not as useful as height relative to the ground
# I need to create a ground image used to subtract from the elevation(Z) in order to get the height of the points relative to the ground


# The DEM was provided with the lidar data, I can clip this and extract the elevation of the ground for the entire area. Where the is something on the ground such as a building, the value is estimated using the nearest ground points available. I can then subtract the laser return value by the ground value to get the relative height of the object instead of the absolute height to sea level. This gives a more accurate height value to use with the algorithms.

# In[38]:


dem_path = "F:/Data/Lidar/images/BCVAN_DEM1m_Clip.tif"
img0 = PIL.Image.open(dem_path)


# In[39]:


dem_array = np.asarray(img0)


# In[40]:


dem_img = img0.convert("F")


# In[41]:


#dem_img = img0.convert("RGB")


# In[42]:


np.asarray(dem_img)


# In[43]:


dem_scaler = len(dem_array)-1
print(dem_scaler)


# In[44]:


x_dem_img = (data['X'] - min(data['X']))/(max(data['X']-min(data['X'])))*dem_scaler
y_dem_img = (data['Y'] - min(data['Y']))/(max(data['Y']-min(data['Y'])))*dem_scaler


# In[45]:


x_dem_img


# In[46]:


coord_array_dem = np.array(pd.DataFrame({"X":x_dem_img,"Y":y_dem_img}))


# In[47]:


coord_array_dem


# In[48]:


dem_value = []
for coord in coord_array_dem:
    val = dem_img.getpixel((coord[0],coord[1]))
    dem_value.append(val)


# In[49]:


len(dem_value)


# In[50]:


data['dem_value'] = dem_value


# In[51]:


data


# In[52]:


data['height'] = data['Z'] - data['dem_value']


# In[53]:


data['height'].describe()


# In[54]:


#data with height values added
data.describe()


# In[55]:


#the return ratio attribute is created by dividing the return number by the total return count
#the histogram for this was shown above
data['return_ratio'] = data['return_number']/data['total_returns']


# In[56]:


#data with return_ratio values added
data.describe()


# # Normalization

# #### Data normalized and preprocessed for analysis

# In[57]:


x_normal = (data['X'] - min(data['X']))/(max(data['X']-min(data['X'])))


# In[58]:


y_normal = (data['Y'] - min(data['Y']))/(max(data['Y']-min(data['Y'])))


# In[59]:


z_normal = (data['Z'] - min(data['Z']))/(max(data['Z']-min(data['Z'])))


# In[60]:


height_normal = (data['height'] - min(data['height']))/(max(data['height']-min(data['height'])))


# In[61]:


i_normal = (data['Intensity'] - min(data['Intensity']))/(max(data['Intensity']-min(data['Intensity'])))


# In[62]:


# new dataframe containing all the normalized values is created
df_normal = pd.DataFrame({'X':x_normal,'Y':y_normal,'Z':z_normal,'height':height_normal,'Intensity':i_normal,'return_ratio':data['return_ratio'],'classification':data['classification']})


# In[63]:


df_normal.describe()


# In[64]:


# Plotting normalized looks the same but with the new scale
fig,ax = plt.subplots(figsize = (10,10))
ax.scatter(df_normal['X'],df_normal['Y'],c='black',s=0.01,alpha=0.5)


# In[65]:


df_normal.dtypes


# # Supervised Classification for ground
# ## Classify the ground for supervised classifier using the provided ground points as labels

# In[66]:


df_unclassified = df_normal.query("classification==1")
df_ground = df_normal.query("classification==2")


# In[67]:


df_ground.describe()


# In[68]:


df_unclassified.describe()


# In[69]:


# Plotting the pre classified/labelled ground points for reference
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df_ground['X'],df_ground['Y'],c='black',s=0.01,alpha=0.5)


# In[70]:


# Plotting the pre classified/labelled unclassified points for reference
#it appears that alot of ground points are still labelled as unclassified.
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df_unclassified['X'],df_unclassified['Y'],c='black',s=0.01,alpha=0.5)


# In[71]:


df_normal.describe()


# In[72]:


train,test = train_test_split(df_normal)


# In[73]:


train_features = pd.DataFrame({"Intensity":train['Intensity'],"return_ratio":train['return_ratio'],"height":train['height']})


# In[74]:


train_labels = np.ravel(pd.DataFrame({"classification":train['classification']}))


# In[75]:


test_features = pd.DataFrame({"Intensity":test['Intensity'],"return_ratio":test['return_ratio'],"height":test['height']})


# In[76]:


test_labels = np.ravel(pd.DataFrame({"classification":test['classification']}))


# In[77]:


#creates the model
model = RandomForestClassifier(max_depth=5,random_state=0,n_estimators=50,criterion="entropy",verbose=0,class_weight="balanced")


# In[78]:


# trains the model - fit the train data to the model
starttime = datetime.datetime.now()
model_fit = model.fit(train_features,train_labels)
endtime = datetime.datetime.now()
runtime = endtime-starttime
print(runtime)


# In[79]:


#decisionpath = model_fit.decision_path(test_features)
#print(decisionpath)


# In[80]:


#predict the test data
test_predictions = model_fit.predict(test_features)


# In[81]:


len([i for i in test_predictions if i == 1])


# In[82]:


len([i for i in test_predictions if i != 1])


# In[83]:


model_fit.score(test_features,test_labels)


# In[84]:


#test set results
confusion_matrix(test_labels,test_predictions)


# In[85]:


table = pd.DataFrame({"Intensity":df_normal['Intensity'],"return_ratio":df_normal['return_ratio'],"H":df_normal['height']})


# In[86]:


table_labels = df_normal['classification']


# In[87]:


table_predictions = model_fit.predict(table)


# In[88]:


len([i for i in table_predictions if i == 1])


# In[89]:


len([i for i in table_predictions if i != 1])


# In[90]:


model_fit.score(table,table_labels)


# In[91]:


#full dataset results
print("Confusion Matrix:")
print(str(confusion_matrix(table_labels,table_predictions)))
print("")
print("Accuracy Score:")
print(str(model_fit.score(table,table_labels)))
print("")
print("Runtime: " + str(runtime))


# In[92]:


df_normal['prediction'] = table_predictions.tolist()


# In[93]:


df_normal


# In[94]:


df_normal.query("classification != prediction")


# In[95]:


predicted_ground = df_normal.query("prediction == 2")


# In[96]:


predicted_ground


# In[97]:


#all of the predict ground points were the last point returned which should be expected
predicted_ground.query("return_ratio!=1")


# In[98]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(predicted_ground['X'],predicted_ground['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[99]:


predicted_non_ground = df_normal.query("prediction == 1")


# In[100]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(predicted_non_ground['X'],predicted_non_ground['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[101]:


predicted_non_ground


# In[102]:


#the non ground points remain to be classified
data = predicted_non_ground


# In[103]:


#non ground points that were not the last return, meaning they are above something else
last_remaining_points = data.query("return_ratio!=1")
non_last_remaining_points = data.query("return_ratio==1")


# In[104]:


#non last collected point of each pulse
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(last_remaining_points['X'],last_remaining_points['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[105]:


#last collected point of each pulse
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(non_last_remaining_points['X'],non_last_remaining_points['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# ## Add Imagery Data
# #### 2015 Imagery data was obtained from the City of Vancouver to extract the RGB values
# #### The image was clipped using external software (QGIS, open-source mapping program) to the same area of interest as above
# #### The selected image size is 4084x4084, the lidar data is normalized by 4084 to extract the nearest pixel value(r,g,b) from the image for each point

# In[106]:


image_path = "F:/Data/Lidar/images/BCVANC15_P9_aoiclip.tif"


# In[107]:


img = PIL.Image.open(image_path)


# In[108]:


rgb_img = img.convert("RGB")


# In[109]:


rgb_img


# In[110]:


#this can be used to crop the imagery if we knew the exact coordinates, but I used QGIS to clip the imagery instead
#left,top,right,bottom = 0,0,4084,4084
#rgb_img = img.crop((left,top,right,bottom))


# In[111]:


#import math
#math.sqrt(istat.Stat(rgb_img).count[0])


# In[112]:


#this size aligns the pixels to the lidar points to extract the rgb values for each point
print(img.size)
img_scaler = img.size[0]-1
print(img_scaler)


# In[113]:


#The image origin (top left) is different than the coordinate system of the lidar so the image needs to be flipped for the calculation to align them
rgb_img_flip = PIL.ImageOps.flip(rgb_img)


# In[114]:


#rgb_img.getpixel((2070,2070))


# In[115]:


# rescales the point values to line up with the pixels in the imagery - same idea as normalization
# this is basically reprojecting the coordinates of the lidar points to the coordinates of the image
x_img = (data['X'] - min(data['X']))/(max(data['X']-min(data['X'])))*img_scaler
y_img = (data['Y'] - min(data['Y']))/(max(data['Y']-min(data['Y'])))*img_scaler


# In[116]:


y_img


# In[117]:


coord_array = np.array(pd.DataFrame({"X":x_img,"Y":y_img}))


# In[118]:


coord_array
# locations on the image to read the rgb values


# #### The nearest R,G,B pixel value from the image is extracted for each lidar point and the results are saved as a field in the data frame

# In[119]:


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


# In[120]:


data


# In[121]:


data_normal = data


# #### The R,G,B values are normalized like the rest:

# In[122]:


r_normal = (data['r'] - min(data['r']))/(max(data['r']-min(data['r'])))
g_normal = (data['g'] - min(data['g']))/(max(data['g']-min(data['g'])))
b_normal = (data['b'] - min(data['b']))/(max(data['b']-min(data['b'])))
data_normal['r'] = r_normal
data_normal['g'] = g_normal
data_normal['b'] = b_normal


# In[123]:


data_normal


# #### Additonal testing and improvements

# In[124]:


data = data_normal


# In[125]:


data


# #### Initial Classification (unsupervised) using kmeans clustering
# #### Attempt to classify points into undetermined classes based on the data
# ##### Variables: Height, Intensity, R, G, B

# In[126]:


#features = pd.DataFrame({"R":data['r'],"G":data['g'],"B":data['b'],"H":data['height'],"I":data['Intensity']})
#features = pd.DataFrame({"R":data['r'],"G":data['g'],"B":data['b']})
features = pd.DataFrame({"H":data['height'],"I":data['Intensity'],"return_ratio":data['return_ratio']})


# In[127]:


features


# In[128]:


X1 = np.array(features)


# In[129]:


X1


# In[130]:


# initialize model
kmeancluster = cluster.KMeans(3,init='random',n_init=10)


# In[131]:


# fit to data
starttime = datetime.datetime.now()
k_clusters = kmeancluster.fit(X1)
endtime = datetime.datetime.now()
runtime = endtime-starttime
print(runtime)


# In[132]:


len(k_clusters.labels_)


# In[133]:


print("Number of clusters:" + str(len(np.unique(k_clusters.labels_))))
print("Points clustered: " + str(len([i for i in k_clusters.labels_ if i != -1])))
print("Calinski-Harabsz Score: " + str(calinski_harabasz_score(X1,k_clusters.labels_)))
print("Silhouette Score: " + str(silhouette_score(X1,k_clusters.labels_,sample_size=10000)))
print("Runtime: " + str(runtime))


# In[134]:


# add results to data frame
data['k_cluster'] = k_clusters.labels_


# In[135]:


#rename for clarity again
results = data


# In[136]:


# visualize the classes
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(results['X'],results['Y'],c=results['k_cluster'],s=0.1,alpha=0.5)


# In[137]:


fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(results['X'],results['Y'],results['Z'],c=results['k_cluster'],s=0.01,alpha=1)


# In[138]:


results


# #### Spatial/Distance Clustering

# #### This section will attempt to use the previous classificiation label to cluster the points into local distinct objects

# In[139]:


df2 = pd.DataFrame({"X":data['X'],"Y":data['Y'],"K":data['k_cluster']})
#df2 = pd.DataFrame({"X":data['X'],"Y":data['Y'],"Z":data['height'],"r":data['r'],"g":data['g'],"b":data['b'],"I":data['Intensity']})
#df2 = pd.DataFrame({"X":data['X'],"Y":data['Y'],"H":data['height'],"I":data['Intensity']})
#df2 = pd.DataFrame({"H":data['height'],"I":data['Intensity']})


# In[140]:


X2 = np.array(df2)


# In[141]:


X2


# In[142]:


# to determine the value for the 'eps' parameter, the value of the inflection point on this distance graph is used
# this is the average distance between the points
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X2)
distances,indices = nbrs.kneighbors(X2)
distances = np.sort(distances,axis=0)
distances = distances[:,1]
plt.plot(distances)


# In[143]:


#create the model
dbscan = cluster.DBSCAN(eps=0.006,min_samples=50,algorithm="auto",leaf_size=30,n_jobs=-1)


# In[144]:


# fit the data to the model to get the cluster id's
starttime = datetime.datetime.now()
db_clusters = dbscan.fit(X2)
endtime = datetime.datetime.now()
runtime = endtime-starttime


# In[145]:


cluster_count = str(len(np.unique(db_clusters.labels_)))
points_clustered = str(len([i for i in db_clusters.labels_ if i != -1]))
print("Number of clusters: " + cluster_count)
print("Points clustered: " + points_clustered)
labels = db_clusters.labels_
print("Calinski-Harabsz Score: " + str(calinski_harabasz_score(X2,labels)))
print("Silhouette Score: " + str(silhouette_score(X2,db_clusters.labels_,sample_size=20000)))
print("Total Runtime: "+str(runtime))


# In[146]:


#add cluster results to dataframe
results['db_cluster'] = list(db_clusters.labels_)


# In[147]:


#filter out results not clustered
results2 = results.query("db_cluster!=-1")


# In[148]:


clrs = list(results2['db_cluster'])


# In[149]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(results2['X'],results2['Y'],c=clrs,s=0.1,alpha=0.5)


# In[150]:


fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(results2['X'],results2['Y'],results2['Z'],c=results2['db_cluster'],s=0.01,alpha=1)


# ### Filtering Results by cluster size

# In[151]:


#looking at individual clusters which each should be equivalent to a unique/distinct "object"
result = results2.query("db_cluster>=0&db_cluster<={0}".format(cluster_count))
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(predicted_ground['X'],predicted_ground['Y'],c='grey',s=10,alpha=1)
ax.scatter(result['X'],result['Y'],c=result['db_cluster'],s=5,alpha=1)


# In[152]:


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

# In[153]:


#imagery again for reference
rgb_img


# In[154]:


result = results2.query("db_cluster>=0 & db_cluster<={0}".format(cluster_count))
fig,ax = plt.subplots(figsize = (25,25))
ax.imshow(rgb_img, extent=[0, 1, 0, 1])
ax.scatter(result['X'],result['Y'],c=result['db_cluster'],s=0.5,alpha=1)
#ax.scatter(predicted_ground['X'],predicted_ground['Y'],c='black',s=10,alpha=1)


# In[155]:


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


# In[156]:


print(" small: " + str(len(small)) + " medium: " + str(len(medium)) + " large: " + str(len(large)))


# In[157]:


alluniqueclusters = result['db_cluster'].unique()


# In[158]:


#small clusters
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


# In[159]:


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


# In[160]:


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


# In[161]:


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


# In[162]:


b.scatter(predicted_ground['X'],predicted_ground['Y'],c='black',s=15,alpha=1)


# In[163]:


a


# In[164]:


result


# In[165]:


#export the results to csv
result.to_csv("F:/Data/Lidar/DataFrame_Clusters.csv")

