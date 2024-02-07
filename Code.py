
"""
@author: Alexandre Cogordan
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import time
import skimage

from sklearn import preprocessing, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier 
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.applications import xception

from skimage import exposure
from skimage.feature import hog


###########################
#    PARAMETER SECTION    #
###########################

size = 32
masking = False #fait
feature_reduction = False # fait
cross_val = 0
model_type = 1
test_proportion = 0.2 #fait
stratisfy_status = True

# Model types available

# 0: random forest
# 1: decision tree
# 2: ada boosting
# 3: gradient boosting
# 4: MLP (multilayer perceptron neural network)

###########################
#      MASK FUNCTION      #
###########################

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([0,0,250])
    upper_hsv = np.array([250,255,255])
    
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

#image segmentation function
def segment_image(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output / 255

#sharpen the image
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

###########################
#        LOAD DATA        #
###########################

#TRAINING - Storing the training images and labelling them into numpy arrays
train_images = []
train_labels = []

if stratisfy_status:
    stratisfy_status = train_labels
else:
    stratisfy_status = None
 
for directory_path in glob.glob("Full_Data/*"):
    training_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, (cv2.IMREAD_COLOR))       
        if img is not None:
            img = cv2.resize(img, (size, size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if masking:
                #masking + segmentation
                img = segment_image(img)
                #sharpening
                img = sharpen_image(img)
                #img = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
            train_images.append(img)
            train_labels.append(training_label)
                    
train_images = np.array(train_images)
train_labels = np.array(train_labels)

#Although we already split it above, we're re-assigning variables to use the convential variable names
x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = test_proportion) # , shuffle = True, stratify = stratisfy_status
#In this case, x variables store the numpy image arrays and the y variables store their labels

#Visualisation of the number of photos per type
def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
     
    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
     
    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'
         
    xtemp = np.arange(len(unique))
     
    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)
 
plt.suptitle('relative amount of photos per type')
plot_bar(y_train, loc='left')
plot_bar(y_test, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train)), 
    'test ({0} photos)'.format(len(y_test))
]);

#Encode labels from text to integers so that the machine understands
le = preprocessing.LabelEncoder()
le.fit(y_test)
y_test_raw = y_test
y_test = le.transform(y_test)
le.fit(y_train)
y_train = le.transform(y_train)
#What this does is change the "Fire" label to 0 and the "Non_Fire" label to 1

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network - we're using a table system to differentiate between the labels
#This is the conventional way of handling labels in neural networks

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


###########################
#    FEATURE REDUCTION    #
###########################

#Alternative to feature selection - feature reduction  
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

if feature_reduction:    
    # create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()
     
    # call fit_transform on each transform converting X_train step by step
    x_train = scalify.fit_transform(hogify.fit_transform(grayify.fit_transform(x_train)))
    
###########################
#           CNN           #
###########################

start_ts=time.time()

activation = 'sigmoid'

model = Sequential()
model.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (size, size, 3)))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

#Flattening our 64 dimension layer to one layer (one column) 
model.add(Flatten())

#Add layers for deep learning prediction
x = model.output  
x = Dense(128, activation = activation, kernel_initializer = 'he_uniform')(x)
prediction_layer = Dense(2, activation = 'softmax')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=model.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary()) 
#What comes in are the images, what comes out are the labels (either 0 - fire or 1 - non_fire)

#Train the CNN model
history = cnn_model.fit(x_train, y_train_one_hot, epochs=7, validation_data = (x_test, y_test_one_hot))    

print("CV Runtime:", time.time()-start_ts)   

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


prediction_NN = cnn_model.predict(x_test) #Shows us the "probabilities" our model suggested for each image in x_test
prediction_NN = np.argmax(prediction_NN, axis=-1) #Converting the values to the final category chosen (1 and 0 format)
prediction_NN = le.inverse_transform(prediction_NN) #Converting the computer format to our categorical format (0/1 to fire/non_fire)

#Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test_raw, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True, yticklabels=['FIRE', 'NON_FIRE'], xticklabels=['FIRE', 'NON_FIRE'])
plt.title('Fire Confusion Matrix');
plt.xlabel('True Class');
plt.ylabel("Predicted Class");

#Below, we can check results of specific images - Neural Network
n=299  #Index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
prediction = np.argmax(cnn_model.predict(input_img))
prediction = le.inverse_transform([prediction])
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", y_test_raw[n])

#for i in range(20,100):
#    input_img = np.expand_dims(img, axis=0)
#    prediction = np.argmax(cnn_model.predict(input_img))
#    prediction = le.inverse_transform([prediction])
#    
#    if(prediction != y_test_raw[i]):    
#        print("The prediction for this image is: ", prediction)
#        print("The actual label for this image is: ", y_test_raw[i])
#        plt.imshow(x_test[i])
#        break


################################
#       DT / RF / BOOSTING     #
################################

#Now, let us use features from convolutional network for decision trees / random forests/ boosting methods
x_for_model = model.predict(x_train)
#X_for_model = model.predict(x_train_prepared)

start_ts=time.time()

if model_type == 0:
    continuing_model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = None, min_samples_split = 3, random_state = 42) 
elif model_type == 1:
    continuing_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=5, min_samples_leaf=1, max_features=None, random_state = 42)
elif model_type == 2:    
    continuing_model = GradientBoostingClassifier(n_estimators = 100, loss = 'deviance', learning_rate = 0.1, max_depth =3, min_samples_split = 3, random_state = 42)
elif model_type == 3:
    continuing_model = AdaBoostClassifier(n_estimators = 100, base_estimator = None, learning_rate = 0.1, random_state = 42)
else:
    continuing_model = MLPClassifier(activation = 'logistic', solver = 'adam', alpha = 0.0001, max_iter = 1000, hidden_layer_sizes = (10,), random_state = 42)

if cross_val == 0:
    # Train the model on training data
    continuing_model.fit(x_for_model , y_train)

    X_test_feature = model.predict(x_test) #Feature extraction
    prediction = continuing_model.predict(X_test_feature) #Prediction using the trained model 
    prediction = le.inverse_transform(prediction) #Converting the computer format to our categorical format (0/1 to fire/non_fire)

    print("Accuracy = ", metrics.accuracy_score(y_test_raw, prediction))
    #print("Accuracy = ", classification_report(y_test, prediction))
    #scores_ACC = continuing_model.score(x_test, y_test)
    #print('Acc:', scores_ACC)
    #scores_AUC = metrics.roc_auc_score(y_test, continuing_model.predict_proba(x_test)[:,1])
    #print('AUC:', scores_AUC)   

else: #cross_val == 1
    #Setup Crossval classifier scorers
    scorers = {'Accuracy': 'accuracy', 'roc_auc':'roc_auc'}
    
    #SciKit Decision Tree - Cross Val
    scores = cross_validate(continuing_model, train_images, train_labels, scoring=scorers, cv=10)

    scores_Acc = scores['test_Accuracy']
    print("Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC = scores['test_roc_auc']                                                                      #Only works with binary classes, not multiclass
    print("AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)

print("CV Runtime:", time.time()-start_ts)   

#Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test_raw, prediction)
sns.heatmap(cm, annot=True, yticklabels=['FIRE', 'NON_FIRE'], xticklabels=['FIRE', 'NON_FIRE'])
plt.title('Fire Confusion Matrix');
plt.xlabel('True Class');
plt.ylabel("Predicted Class");

#Below, we can check results of specific images - Random Forest
n=42 #Index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_features=model.predict(input_img)
prediction = continuing_model.predict(input_img_features)[0] 
prediction = le.inverse_transform([prediction])
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", y_test_raw[n])
