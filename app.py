from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st
from PIL import Image
import heapq as hq

@st.cache(allow_output_mutation=True)
def get_model():
        model = load_model('inception.hdf5')
        print('Model Loaded')
        return model 

        
def predict(image):
        loaded_model = get_model()

        image = image.resize((224, 224))
        image = np.asarray(image)
        image = image/255.0
        image = np.reshape(image,[1,224,224,3])
        #st.write("{}".format(image.shape))
        classes = loaded_model.predict(image)#.argmax()

        ans = []
        for i, n in enumerate(classes[0]):
          hq.heappush(ans, (n, i))
          if len(ans)>5:
            hq.heappop(ans)
        return ans
sign_names = {0: 'adonis', 1: 'american snoot', 2: 'an 88', 3: 'banded peacock', 4: 'beckers white', 5: 'black hairstreak', 6: 'cabbage white', 7: 'chestnut', 8: 'clodius parnassian', 9: 'clouded sulphur',
                  10: 'copper tail', 11: 'crecent', 12: 'crimson patch', 13: 'eastern coma', 14: 'gold banded', 15: 'great eggfly', 16: 'grey hairstreak', 17: 'indra swallow', 18: 'julia', 19: 'large marble',
                  20: 'malachite', 21: 'mangrove skipper', 22: 'metalmark', 23: 'monarch', 24: 'morning cloak', 25: 'orange oakleaf', 26: 'orange tip', 27: 'orchard swallow', 28: 'painted lady', 29: 'paper kite', 
                  30: 'peacock', 31: 'pine white', 32: 'pipevine swallow', 33: 'purple hairstreak', 34: 'question mark', 35: 'red admiral', 36: 'red spotted purple', 37: 'scarce swallow', 38: 'silver spot skipper', 39: 'sixspot burnet',
                  40: 'skipper', 41: 'sootywing', 42: 'southern dogface', 43: 'straited queen', 44: 'two barred flasher', 45: 'ulyses', 46: 'viceroy', 47: 'wood satyr', 48: 'yellow swallow tail', 49: 'zebra long wing'}

st.title("Butterfly Species Classifier")
st.write('This app takes in an image of a butterfly and gives its species.\nThe model has been trained by on 50 different species of butterflies.')
st.write('')
st.write("[This](https://www.kaggle.com/pnkjgpt/butterfly-classification-dataset) is the dataset that it has been trained on and [this](https://www.kaggle.com/pnkjgpt/multiclass-image-classification-transfer-learning) is the training notebook")
st.write('')

uploaded_file = st.file_uploader("Choose an image of a butterfly whose species you want to find", type=["jpg", "png", "jpeg"])
if uploaded_file:
        image = Image.open(uploaded_file)
        st.write("")
        st.image(image, caption='Your Uploaded Image', use_column_width=False,)
        st.write("")
        if st.button('Predict'):
                st.write("The given image has been classified as:")
                label = predict(image)
                #res = sign_names.get(label)
                for p, c in sorted(label, reverse = True):
                  st.write('{0} : {1}%'.format(sign_names[c], round(p*100, 3)))
