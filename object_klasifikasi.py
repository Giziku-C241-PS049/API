import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

model = None

def load_model():
  model = tf.keras.models.load_model('./model_mobile_valacc_0.9194.h5')
  return model

def predict_image(image: Image.Image):
  global model
  if model is None:
    model = load_model()
    
  image = np.asarray(image.resize((224, 224)))[..., :3]
  image = np.expand_dims(image, 0)
  image = image / 127.5 - 1.0
  
  class_probabilities = model.predict(image) 

  class_names = ['anggur', 'apel', 'ayam_betutu', 'ayam_goreng', 'ayam_pop', 'bakso', 'batagor', 'burger', 'cherry', 'cireng', 'coto_makassar', 'dendeng', 'gudeg', 'gulai_ikan', 'jeruk', 'kerak_telor', 'kiwi', 'mangga', 'mie_aceh', 'nasi_goreng', 'nasi_kuning', 'nasi_padang', 'nasi_pecel', 'pempek', 'pisang', 'rawon', 'rendang', 'sate', 'sawo', 'serabi', 'soto', 'strawberi', 'tahu_sumedang', 'telur_balado', 'telur_dadar']

  predicted_classes_index = np.argmax(class_probabilities)
  predicted = class_names[predicted_classes_index]
  return predicted
    
def read_imagefile(file) -> Image.Image:
  image = Image.open(BytesIO(file))
  return image
