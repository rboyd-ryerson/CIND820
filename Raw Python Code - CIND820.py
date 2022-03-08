#!/usr/bin/env python
# coding: utf-8

# ## CIND820 Project Course Code
# #### Ryan Boyd - 501072988
# ### Assignment 3
# ### Initial Results and Code

# In[1]:


#import all libraries
import laspy as lp, sklearn as skl, numpy as np, matplotlib as mp, pandas as pd


# In[2]:


from sklearn import cluster
from sklearn import preprocessing as prep
from sklearn.neighbors import NearestNeighbors


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import PIL
from PIL import ImageStat as istat
from PIL import ImageOps


# #### Data is loaded and prepared

# In[5]:


#original dataset https://nrs.objectstore.gov.bc.ca/gdwuts/092/092g/2016/dsm/bc_092g025_3_4_2_xyes_8_utm10_20170601_dsm.laz
#renamed here for clarity
path_to_data = "F:/Data/Lidar/dtvan/dtvan.laz"
with lp.open(path_to_data) as las_file:
    las_data = las_file.read()


# In[6]:


# data loaded into a dataframe
df = pd.DataFrame({"X":list(las_data.x),"Y":list(las_data.y),"Z":list(las_data.z),"Intensity":las_data.intensity,"return_num":las_data.return_number,"totalreturns":las_data.num_returns,"classification":las_data.classification})


# In[7]:


#full dataset displayed on a scatter plot
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df['X'],df['Y'],zorder=1,alpha=0.25,c='black',s=0.001)


# In[8]:


print("Total points:" + str(las_data.header.point_count))


# In[9]:


print("Classes: " + str(set(list(df['classification']))))


# In[10]:


#data summary
df.describe()


# #### The full original dataset is too large to work with so it is clipped to a smaller study area. The dataframe is queried by the study area longitude and latitude boundary. The pre-classified ground points, class 2, are removed (since we are not concerned with ground right now), class 1 are the unclassified points and we only want to work with these.

# In[11]:


# Define the area of interest, these values in meters are in the "NAD83 UTM 10" coordinate system of the provided dataset
# These are the upper and lower limits in meters to be used, these can be found using google maps or other free sites/software
# These were selected somewhat arbitrarily
aoi_extent = {'xmax':492349.0731766,'xmin':492043.6935073,'ymax':5458645.8660691,'ymin':5458340.4864470}


# In[12]:


#query the dataframe to return only the points within the extent above and remove the points defined as ground as well
df_clip = df.query("X>{0}&X<{1}&Y>{2}&Y<{3}&classification==1".format(aoi_extent['xmin'],aoi_extent['xmax'],aoi_extent['ymin'],aoi_extent['ymax']))


# In[13]:


df_clip


# In[14]:


#study area points displayed on a scatter plot
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(df_clip['X'],df_clip['Y'],zorder=1,alpha=0.5,c='black',s=0.01)


# In[15]:


#the height is used value to visualize in 3d, since the values are in meters for all 3 axes, it plots nicely as is
fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(df_clip['X'],df_clip['Y'],df_clip['Z'],c='black',s=0.01,alpha=0.5)

#from matplotlib import cm
#ax.plot_surface(df_clip['X'],df_clip['Y'],df_clip['Z'],cmap=cm.coolwarm,linewidth=0,antialiased=False)


# In[16]:


# The Z values are an absolute elevation which is not as useful as height relative to the ground
df_ground = df.query("X>{0}&X<{1}&Y>{2}&Y<{3}&classification==2".format(aoi_extent['xmin'],aoi_extent['xmax'],aoi_extent['ymin'],aoi_extent['ymax']))


# In[17]:


df_ground['Z']
# I need to create a ground image used to subtract from the elevation(Z) in order to get the height of the points relative to the ground


# In[18]:


# Plotting the pre classified/labelled ground points for reference
fig,ax = plt.subplots(figsize = (10,10))
ax.scatter(df_ground['X'],df_ground['Y'],c='black',s=0.01,alpha=0.5)


# #### Data normalized and preprocessed for analysis

# In[19]:


x_normal = (df_clip['X'] - min(df_clip['X']))/(max(df_clip['X']-min(df_clip['X'])))


# In[20]:


y_normal = (df_clip['Y'] - min(df_clip['Y']))/(max(df_clip['Y']-min(df_clip['Y'])))


# In[21]:


z_normal = (df_clip['Z'] - min(df_clip['Z']))/(max(df_clip['Z']-min(df_clip['Z'])))


# In[22]:


i_normal = (df_clip['Intensity'] - min(df_clip['Intensity']))/(max(df_clip['Intensity']-min(df_clip['Intensity'])))


# In[23]:


# new dataframe containing all the normalized values is created
df_normal = pd.DataFrame({'X':x_normal,'Y':y_normal,'Z':z_normal,'I':i_normal,'return_num':df_clip['return_num'],'total_returns':df_clip['totalreturns']})


# In[24]:


df_normal


# In[25]:


# Plotting normalized looks the same but with the new scale
fig,ax = plt.subplots(figsize = (10,10))
ax.scatter(df_normal['X'],df_normal['Y'],c='black',s=0.01,alpha=0.5)


# ## Add Imagery Data
# #### 2015 Imagery data was obtained from the City of Vancouver to extract the RGB values
# #### The image was clipped using external software (QGIS, open-source mapping program) to the same area of interest as above
# #### The selected image size is 4084x4084, the lidar data is normalized by 4084 to extract the nearest pixel value(r,g,b) from the image for each point

# In[26]:


image_path = "F:/Data/Lidar/images/BCVANC15_P9_aoiclip.tif"


# In[27]:


img = PIL.Image.open(image_path)


# In[28]:


rgb_img = img.convert("RGB")


# In[29]:


rgb_img


# In[30]:


#this can be used to crop the imagery if we knew the exact coordinates, but I used QGIS to clip the imagery instead
#left,top,right,bottom = 0,0,4084,4084
#rgb_img = img.crop((left,top,right,bottom))


# In[31]:


#import math
#math.sqrt(istat.Stat(rgb_img).count[0])


# In[32]:


#this size aligns the pixels to the lidar points to extract the rgb values for each point
img.size


# In[33]:


#The image origin (top left) is different than the coordinate system of the lidar so the image needs to be flipped for the calculation to align them
rgb_img = PIL.ImageOps.flip(rgb_img)


# In[34]:


#rgb_img.getpixel((2070,2070))


# In[35]:


# rescales the point values to line up with the pixels in the imagery - same idea as normalization
# this is basically reprojecting the coordinates of the lidar points to the coordinates of the image
x_img = (df_clip['X'] - min(df_clip['X']))/(max(df_clip['X']-min(df_clip['X'])))*4083
y_img = (df_clip['Y'] - min(df_clip['Y']))/(max(df_clip['Y']-min(df_clip['Y'])))*4083


# In[36]:


y_img


# In[37]:


coord_array = np.array(pd.DataFrame({"X":x_img,"Y":y_img}))


# In[38]:


coord_array
# locations on the image to read the rgb values


# #### The nearest R,G,B pixel value from the image is extracted for each lidar point and the results are saved as a field in the data frame

# In[39]:


rgb_data = []
rgb_data_r = []
rgb_data_g = []
rgb_data_b = []
for coord in coord_array:
    rgb=rgb_img.getpixel((coord[0],coord[1]))
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    rgb_data.append(rgb)
    rgb_data_r.append(r)
    rgb_data_g.append(g)
    rgb_data_b.append(b)
df_normal['rgb'] = rgb_data
df_normal['r'] = rgb_data_r
df_normal['g'] = rgb_data_g
df_normal['b'] = rgb_data_b


# In[40]:


df_normal


# #### The R,G,B values are normalized like the rest:

# In[41]:


r_normal = (df_normal['r'] - min(df_normal['r']))/(max(df_normal['r']-min(df_normal['r'])))
g_normal = (df_normal['g'] - min(df_normal['g']))/(max(df_normal['g']-min(df_normal['g'])))
b_normal = (df_normal['b'] - min(df_normal['b']))/(max(df_normal['b']-min(df_normal['b'])))
df_normal['r'] = r_normal
df_normal['g'] = g_normal
df_normal['b'] = b_normal


# In[42]:


df_normal


# #### Dataset statistics and information - exploratory

# In[43]:


#renaming the data frame for clarity
data = df_normal


# In[44]:


mp.pyplot.hist(data['X'])


# In[45]:


mp.pyplot.hist(data['I'])


# In[46]:


mp.pyplot.hist(data['r'])


# In[47]:


#summarize the normalized data
data.describe()


# #### Additonal testing and improvements

# In[ ]:





# In[ ]:





# #### Initial Classification (unsupervised) using kmeans clustering
# #### Attempt to classify points into undetermined classes based on the data
# ##### Variables: Height, Intensity, R, G, B

# In[48]:


features = pd.DataFrame({"R":df_normal['r'],"G":df_normal['g'],"B":df_normal['b'],"Z":df_normal['Z'],"I":df_normal['I']})


# In[49]:


features


# In[50]:


X1 = np.array(features)


# In[51]:


X1


# In[52]:


# initialize model
kmeancluster = cluster.KMeans(5,init='random',n_init=10)


# In[53]:


# fit to data
k_clusters = kmeancluster.fit(X1)


# In[54]:


len(k_clusters.labels_)


# In[ ]:


print("Number of clusters:" + str(len(np.unique(k_clusters.labels_))))
print("Points clustered: " + str(len([i for i in k_clusters.labels_ if i != -1])))


# In[ ]:


results = k_results.query("db_cluster!=-1")


# In[55]:


# add results to data frame
data['k_cluster'] = k_clusters.labels_


# In[56]:


#rename for clarity again
results = data


# #### Visualization

# In[57]:


# visualize the classes
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(results['X'],results['Y'],c=results['k_cluster'],s=0.1,alpha=0.5)


# In[58]:


fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(data['X'],data['Y'],data['Z'],c=data['k_cluster'],s=0.01,alpha=1)


# #### Spatial/Distance Clustering

# #### This section will attempt to use the previous classificiation label to cluster the points into local distinct objects

# In[59]:


df2 = pd.DataFrame({"X":data['X'],"Y":data['Y'],"k_cluster":data['k_cluster']})


# In[60]:


X2 = np.array(df2)


# In[61]:


X2


# In[62]:


# to determine the value for the 'eps' parameter, the value of the inflection point on this distance graph is used
# this is the average distance between the points
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X2)
distances,indices = nbrs.kneighbors(X2)
distances = np.sort(distances,axis=0)
distances = distances[:,1]
plt.plot(distances)


# In[63]:


dbscan = cluster.DBSCAN(eps=0.005,min_samples=50,algorithm="auto",leaf_size=15,n_jobs=-1)


# In[64]:


db_clusters = dbscan.fit(X2)


# In[65]:


print("Number of clusters:" + str(len(np.unique(db_clusters.labels_))))
print("Points clustered: " + str(len([i for i in db_clusters.labels_ if i != -1])))


# In[66]:


#add cluster results to dataframe
results['db_cluster'] = db_clusters.labels_


# In[67]:


#filter out results not clustered
results2 = results.query("db_cluster!=-1")


# #### Visualization

# In[68]:


fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(results2['X'],results2['Y'],c=results2['db_cluster'],s=0.1,alpha=0.5)


# In[69]:


fig,ax = plt.subplots(figsize = (15,15)),plt.axes(projection='3d')
ax.scatter3D(results2['X'],results2['Y'],results2['Z'],c=results2['db_cluster'],s=0.01,alpha=1)


# In[113]:


#looking at individual clusters which each should be equivalent to a unique/distinct "object"
result = results2.query("db_cluster<1000&db_cluster>500")
fig,ax = plt.subplots(figsize = (15,15))
ax.scatter(result['X'],result['Y'],c=result['db_cluster'],s=10,alpha=1)


# ##### Testing Alternative models

# In[70]:


#optics = cluster.OPTICS(min_samples=10,max_eps=2,leaf_size=15)


# In[71]:


#op_clusters = optics.fit(X2)


# In[72]:


#len(set(db_clusters.labels_))
#results['op_clusters'] = op_clusters.labels_


# #### Analysis of results

# The elevation has a large effect on the classification and the R,G,B values from imagery causes the shadows to be very apparent, I may need to adjust the weighting of these values to mitigate issues. I will need to look into neighbourhood characteristics of groups of several points such as flat surfaces or planes and sharp edges, I did read about this in the research papers but it appears to be somewhat technically intensive for my purposes but I will investigate.
# 
# 
# 
# Next Steps:
# -Optimize value of K for K-means, (elbow method or similar)
# -Update the Z values to be a relative height to the ground as opposed to an absolute or (mean sea level) height.
# -Optimize the parameters to improve results (eps,min_samples,leaf_size,etc.)
# -Switch orthophoto for one without shadows
# -Find method to extract clusters as features
# -Create data for known features to compare
# -Find method to compare results to known features
# -Determine the evaluation criteria and minimum viable product
# -Evaluate performance of models and project results overall
