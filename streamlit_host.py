import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import imutils
import numpy as np
from tensorflow.keras.preprocessing import image


st.title("Brain Tumor Detection")
st.write("By Abhay | Akash | Aman | Himanshu")

st.write("NSUT BTP Project-2022")
uploaded_file = st.file_uploader("Choose a image file", type="jpg")



# *******************************************************Models************************************************************#
prediction_model = tf.keras.models.load_model("./Detection/detection_model.hdf5")
classification_model = tf.keras.models.load_model("./Classification/classification_model.hdf5")
# *************************************************************************************************************************#





# ************************************************Decoding Image Data************************************************************#
if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

# *************************************************************************************************************************#






# ************************************************** PreProcessing **************************************************#
    resized = cv2.resize(opencv_image,(224,224))

    image=resized
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    resized = mobilenet_v2_preprocess_input(resized)
    image224 = resized[np.newaxis,...]    
    
    image240 = cv2.resize(new_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(new_image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = image / 255.

#     image = image.reshape((1, 240, 240, 3))
    
    test_image = image
#     test_image = image.img_to_array(test_image)
    test_image=test_image/255
    test_image = np.expand_dims(test_image, axis = 0)

# *************************************************************************************************************************#
    




# ****************************************************Display**************************************************************#   
    st.image(image240, channels="RGB")
# *************************************************************************************************************************#
    




# *****************************************************Button**************************************************************#
    Genrate_pred = st.button("Generate Prediction")    
# *************************************************************************************************************************#
    




# *****************************************************Prediction***********************************************************#    
    if Genrate_pred:       
        
        res = prediction_model.predict(test_image)
        
        if(res > 0):
            st.title("No Tumor is Present")
        else:
            
            #prediction resulted from classification_model
            prediction = classification_model.predict(image224).argmax()

            #category resulted from classification_model            
            class_dict = {0: "glioma_tumor", 1: "meningioma_tumor", 2: "pituitary_tumor"}
            curr_cat=class_dict [prediction]
            
            st.title("Tumor is Present and category is {}".format(curr_cat))

# *************************************************************************************************************************#





            
#********************************************Things for showing region of tumor********************************************#
            img=uploaded_file
            Img = opencv_image
            # curImg = np.array(img)
            curImg=opencv_image

            gray = cv2.cvtColor(np.array(Img), cv2.COLOR_BGR2GRAY)
            [ret, thresh] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            curImg = opening

            # sure background area
            sure_bg = cv2.dilate(curImg, kernel, iterations=3)

            # Finding sure foreground area
            # curImg = image.img_to_array(curImg, dtype='uint8')
            dist_transform = cv2.distanceTransform(curImg, cv2.DIST_L2, 5)
            [ret, sure_fg] = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Find unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg,dtype=cv2.CV_32F)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now mark the region of unknown with zero
            markers[unknown == 255] = 0   
            markers = cv2.watershed(opencv_image, markers)

            opencv_image[markers == -1] = [255, 0, 0]

            tumorImage = cv2.cvtColor(opencv_image, cv2.COLOR_HSV2BGR)
            curImg = tumorImage            
            st.image(curImg)
# *************************************************************************************************************************#
