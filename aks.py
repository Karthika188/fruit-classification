import numpy as np
from keras.models import load_model
from keras.preprocessing import image



model = load_model('model.h5')

# summarize model
model.summary()
imagename = "56_100.jpg"
test_image = image.load_img(imagename, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(result)
# if result[0][0] == 1:
#     prediction = 'dog'
#     return [{"image": prediction}]
# elif result[0][0] == 1:
#     prediction = 'dog'
#     return [{"image": prediction}]
# else:
#     prediction = 'cat'
#     return [{"image": prediction}]