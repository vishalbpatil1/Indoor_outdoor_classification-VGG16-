'''run on Anaconda prompt streamlit run app.py'''
import streamlit as st
import pandas as pd
import numpy  as np
import cv2
import os
from PIL import Image
from keras.preprocessing import image
import tensorflow as tf
font = cv2.FONT_HERSHEY_PLAIN
font_=cv2.FONT_HERSHEY_COMPLEX
def text_display(text,text_offset_x,text_offset_y,color):
    (text_width, text_height) = cv2.getTextSize(text, font_, fontScale=1, thickness=1)[0]
    box_coords = ((text_offset_x, text_offset_y+20), (text_offset_x + text_width + 10, text_offset_y - text_height-20))
    cv2.rectangle(image_frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
    cv2.putText(image_frame, text, (text_offset_x, text_offset_y), font_, fontScale=1, color=color, thickness=2)
    return image_frame


model =tf.keras.models.load_model('C:\\Users\\Vishal\\scens_indoor_outdoor\\model\\model_1.h5')
st.markdown(f'''<center><h1 style="font-family:cursive;color:rgb(0,250,200);text-decoration-line:overline underline;text-decoration-style:double ;">VisionNLP</h1></center>''',unsafe_allow_html=True)

st.markdown(f'''<center><h1 style="background-color:DodgerBlue ;">Real Time indoor outdoor detection</h1><center>''',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose image from your folders ",type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
	image_scens= Image.open(uploaded_file)
	image_scens = image_scens.resize((700,700), Image.ANTIALIAS)
	img_array = image.img_to_array(image_scens)
	image.save_img('C:\\Users\\Vishal\\scens_indoor_outdoor\\model\\test1.jpg', img_array)
	st.image(image_scens)
if st.button('proceed'):
	image_frame = cv2.imread('C:\\Users\\Vishal\\scens_indoor_outdoor\\model\\test1.jpg')
	face = cv2.resize(image_frame,(224,224))
	xx = image.img_to_array(face)
	xx = np.expand_dims(xx, axis = 0)
	xx /= 255
	pred = model.predict(xx)
	target_name=['indoor','outdoor']	
	result = target_name[pred.argmax()]
	text1 = str('Image class : ')+str(result)
	text2 = 'Probability :  %.3f'% np.max(pred)
	text_display(text1,140,600,color=(200,200,0))
	text_display(text2,160,650,color=(200,200,200))
	cv2.imwrite('model\\output.jpg',image_frame)
	image_output= image.load_img('model\\output.jpg',target_size=(700,700))
	st.image(image_output) 
	st.write(str(result))
