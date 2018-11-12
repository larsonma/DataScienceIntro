
# Final Project Proposal

## Dataset of Intrest

For the final project of CE 4981, I will be using the Airbnb price prediction dataset. This dataset is avilable at <a href="https:www.kaggle.com/stevezhenghp/airbnb-price-prediction">https:www.kaggle.com/stevezhenghp/airbnb-price-prediction</a> and can be downloaded as a plain csv file.

## Objectives

### Main Objectives
The objective of this project is to use the abundant information to help create a model for predicting the prices of airbnb housing. In addition to this, I would like to explore the possibility of predicting the review score for a airbnb house based on it's attributes, which may also be related to the the price that an owner of an airbnb house can set. 

While those objectives are good for approaches using linear regression for supervised learning, it may also be helpful to perform unsupervised learning on the dataset to help visualize clusters of airbnbs. It may be possible that airbnb rooms are priced or rated in ways in which they can be categorized into groups; performing cluster analysis will help identify these groups, and the common characteristics that have the largest impact on this. 

### Fine-tuned Objectives 
Fine-tuned objectives relate to identifying more subtle, yet important and intersting relationships in the data, which in the end may have a significant impact on the pricing of an airbnb. These may include exploring the following hypothesis:

* analyzing the pricing of airbnb prices based on geographical location, and determining if paying a premium for pricing correlates to a better overall experience, which is reflected by a user's review of the airbnb. 

* determine if there is a relation between the length of the description + attributes of the host, and the quality of the airbnb (which can again be determined based on the pricing and the reviews of the airbnb). I have a theory that when the description is longer, it correlates to a more dedicated host, which likely has a positive correlation possibly with pricing, and likely with reviews. 

* Determine if the number of amenities correlates stronger with the number of bedrooms, bathrooms, and the accommodation capacity than the price of the airbnb (or vice-versa). This may indicate whether amenities are a function of size (you can fit more things in a larger space), or if amenities are considered to be a luxurious item (which would mean a higher price). It's possible that neither has a stronger correlation and rather amenities are a function of both size and luxury. *Note*: this might end up being difficult to determine because size likely has a positive correlation with price. This may need to be explored prior.

* and more...

## Loading the Dataset


```python
import pandas as pd
import numpy as np

airbnb = pd.read_csv("./data/train.csv")
airbnb.head(n=10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>log_price</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>amenities</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bed_type</th>
      <th>cancellation_policy</th>
      <th>cleaning_fee</th>
      <th>...</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>name</th>
      <th>neighbourhood</th>
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>thumbnail_url</th>
      <th>zipcode</th>
      <th>bedrooms</th>
      <th>beds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6901257</td>
      <td>5.010635</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{"Wireless Internet","Air conditioning",Kitche...</td>
      <td>3</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>strict</td>
      <td>True</td>
      <td>...</td>
      <td>40.696524</td>
      <td>-73.991617</td>
      <td>Beautiful brownstone 1-bedroom</td>
      <td>Brooklyn Heights</td>
      <td>2</td>
      <td>100.0</td>
      <td>https://a0.muscache.com/im/pictures/6d7cbbf7-c...</td>
      <td>11201</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6304928</td>
      <td>5.129899</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{"Wireless Internet","Air conditioning",Kitche...</td>
      <td>7</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>strict</td>
      <td>True</td>
      <td>...</td>
      <td>40.766115</td>
      <td>-73.989040</td>
      <td>Superb 3BR Apt Located Near Times Square</td>
      <td>Hell's Kitchen</td>
      <td>6</td>
      <td>93.0</td>
      <td>https://a0.muscache.com/im/pictures/348a55fe-4...</td>
      <td>10019</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7919400</td>
      <td>4.976734</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{TV,"Cable TV","Wireless Internet","Air condit...</td>
      <td>5</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>...</td>
      <td>40.808110</td>
      <td>-73.943756</td>
      <td>The Garden Oasis</td>
      <td>Harlem</td>
      <td>10</td>
      <td>92.0</td>
      <td>https://a0.muscache.com/im/pictures/6fae5362-9...</td>
      <td>10027</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13418779</td>
      <td>6.620073</td>
      <td>House</td>
      <td>Entire home/apt</td>
      <td>{TV,"Cable TV",Internet,"Wireless Internet",Ki...</td>
      <td>4</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>flexible</td>
      <td>True</td>
      <td>...</td>
      <td>37.772004</td>
      <td>-122.431619</td>
      <td>Beautiful Flat in the Heart of SF!</td>
      <td>Lower Haight</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/im/pictures/72208dad-9...</td>
      <td>94117.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3808709</td>
      <td>4.744932</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{TV,Internet,"Wireless Internet","Air conditio...</td>
      <td>2</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>...</td>
      <td>38.925627</td>
      <td>-77.034596</td>
      <td>Great studio in midtown DC</td>
      <td>Columbia Heights</td>
      <td>4</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>20009</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12422935</td>
      <td>4.442651</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>{TV,"Wireless Internet",Heating,"Smoke detecto...</td>
      <td>2</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>strict</td>
      <td>True</td>
      <td>...</td>
      <td>37.753164</td>
      <td>-122.429526</td>
      <td>Comfort Suite San Francisco</td>
      <td>Noe Valley</td>
      <td>3</td>
      <td>100.0</td>
      <td>https://a0.muscache.com/im/pictures/82509143-4...</td>
      <td>94131</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11825529</td>
      <td>4.418841</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{TV,Internet,"Wireless Internet","Air conditio...</td>
      <td>3</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>...</td>
      <td>33.980454</td>
      <td>-118.462821</td>
      <td>Beach Town Studio and Parking!!!11h</td>
      <td>NaN</td>
      <td>15</td>
      <td>97.0</td>
      <td>https://a0.muscache.com/im/pictures/4c920c60-4...</td>
      <td>90292</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13971273</td>
      <td>4.787492</td>
      <td>Condominium</td>
      <td>Entire home/apt</td>
      <td>{TV,"Cable TV","Wireless Internet","Wheelchair...</td>
      <td>2</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>...</td>
      <td>34.046737</td>
      <td>-118.260439</td>
      <td>Near LA Live, Staple's. Starbucks inside. OWN ...</td>
      <td>Downtown</td>
      <td>9</td>
      <td>93.0</td>
      <td>https://a0.muscache.com/im/pictures/61bd05d5-c...</td>
      <td>90015</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>180792</td>
      <td>4.787492</td>
      <td>House</td>
      <td>Private room</td>
      <td>{TV,"Cable TV","Wireless Internet","Pets live ...</td>
      <td>2</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>...</td>
      <td>37.781128</td>
      <td>-122.501095</td>
      <td>Cozy Garden Studio - Private Entry</td>
      <td>Richmond District</td>
      <td>159</td>
      <td>99.0</td>
      <td>https://a0.muscache.com/im/pictures/0ed6c128-7...</td>
      <td>94121</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5385260</td>
      <td>3.583519</td>
      <td>House</td>
      <td>Private room</td>
      <td>{"Wireless Internet","Air conditioning",Kitche...</td>
      <td>2</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>...</td>
      <td>33.992563</td>
      <td>-117.895997</td>
      <td>No.7 Queen Size Cozy Room 舒适大床房</td>
      <td>NaN</td>
      <td>2</td>
      <td>90.0</td>
      <td>https://a0.muscache.com/im/pictures/8d2f08ce-b...</td>
      <td>91748</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 29 columns</p>
</div>



## Partners

I will be performing this project individually.
