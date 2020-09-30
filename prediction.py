#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


# In[15]:


# loading the model
model=load_model('mnist_handwritten.h5')


# In[30]:


# reading the image
img=cv2.imread('3.jpg')

# converting image to greyscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# threshold the gray scale image (so that only the images are white and rest things are black)
ret,thresh=cv2.threshold(gray,75,255,cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

preprocessed_digits = []

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    
    # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
    cv2.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    
    # Cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y+h, x:x+w]
    
    # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(digit, (18,18))
    
    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
    
    # Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_digits.append(padded_digit)

print("\n\n\nContoured Image")
plt.imshow(img, cmap="gray")
plt.show()


# In[32]:


# predicting the digits


for x in preprocessed_digits:
    prediction = model.predict(x.reshape(1, 28, 28, 1))   
    print ("\n\nPREDICTION \n\n")
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.show()
    print("\n\nFinal Output: {}".format(np.argmax(prediction)))


# In[ ]:




