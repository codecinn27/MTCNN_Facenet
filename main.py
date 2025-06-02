from mtcnn import MTCNN #pre trained model used to detet faces in images
from keras_facenet import FaceNet # to generate feature vectors  for face verification
import cv2 #handling image resize, reading and manipulating images
import numpy as np # for handling image data and numerical operations on arrays
import matplotlib.pyplot as plt # for data visualization
from scipy.spatial.distance import cosine # to calculate cosine distance between two vectors
from skimage import io # for loading images from local paths or URLS
import warnings # for cleaner output
warnings.filterwarnings('ignore') #ignore irrelevant warnings
plt.style.use('dark_background') #apply dark theme for plots


image1_cr7 = 'img_dataset\cr7\cr7-face-formal.jpg'
image2_cr7 = 'img_dataset\cr7\cr7-face-red.jpg'

image1_messi = 'img_dataset\messi\messi-blue.jpeg'
image2_messi = 'img_dataset\messi\messi-hair.webp'

image_list = [image1_cr7, image2_cr7, image1_messi, image2_messi]

# Check image is loadeble or not 
# plt.figure(figsize=(15,7))

# for i in range(len(image_list)):
#   plt.subplot(2,2,i+1)
#   plt.suptitle('Images Sample')
#   img = plt.imread(image_list[i])
#   plt.imshow(img)
#   plt.axis('off')
#   plt.title(f" Image (i+1), shape: {img.shape}, size=10")
  
class Color:
  GREEN = '\033[92m'
  BLUE = '\033[94m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  END = '\033[0m'
  
  
def Face_Detection(image= None, model= MTCNN(), color='red', url=None, size=10):
  print(f'{Color.BOLD} The process has been started for detection faces .....')

  plt.style.use('dark_background')
  if url: 
    img = io.imread(url)

  if image:
    img = plt.imread(image)
  
  model = model
  coordinates = model.detect_faces(img)

  plt.style.use('dark_background')

  plt.figure(figsize= (12,6))

  plt.subplot(1,2,1)
  plt.imshow(img)

  plt.title('Face Detection')
  plt.axis('off')
  ax = plt.gca()

  for coordinate in coordinates:
    print('{}The box coordinates : {} {}\n{}The confidence: {} {}\nThe key points: {} {}'.format(Color.GREEN, Color.END, coordinate['box'], Color.RED, Color.END , coordinate['confidence'], Color.BLUE, Color.END, coordinate['keypoints']))

    x,y, width ,height = coordinate['box']
    rect = plt.Rectangle((x,y), width, height, fill=False, color = color)
    ax.add_patch(rect)

  
  plt.subplot(1,2,2)
  plt.imshow(img)
  plt.title(f'key points')
  plt.axis('off')


  for coordinate in coordinates:

    left_eye = coordinate['keypoints']['left_eye']
    right_eye = coordinate['keypoints']['right_eye']
    nose = coordinate['keypoints']['nose']
    mouth_left = coordinate['keypoints']['mouth_left']
    mouth_right = coordinate['keypoints']['mouth_right']

    plt.scatter(left_eye[0], left_eye[1], color='red', s=size)
    plt.scatter(right_eye[0], right_eye[1], color='red', s=size)
    plt.scatter(nose[0], nose[1], color='red', s=size)
    plt.scatter(mouth_left[0], mouth_left[1], color='red', s=size)
    plt.scatter(mouth_right[0], mouth_right[1], color='red', s=size)

  plt.show()
  print(f'{Color.BOLD} The process has been completed .....')
  

def reading_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def Face_Detection_FaceNet(image, margin=0.2):
  model = MTCNN()
  detection = model.detect_faces(image)

  if detection:
    x, y, width , height = detection[0]['box']
    margin = margin
    x_min, y_min = max(x-int(margin*width), 0), max(y-int(margin*height), 0)
    x_max, y_max = min(x+width+int(margin*width), image.shape[1]), min(y+height+int(margin*height), image.shape[0])

    face_box = image[y_min:y_max, x_min:x_max]
    image_with_face_detection = cv2.resize(face_box, (160,160))
    return image_with_face_detection
  
  else:
    return None

def Embedding(fimage_with_face_detection):
  model = FaceNet()
  Embedding_vector = model.embeddings([fimage_with_face_detection])[0]
  return Embedding_vector

def Similarity_Measurement(embedding_vector1, embedding_vector2, threshold=0.7):
  Distance = cosine(embedding_vector1, embedding_vector2)
  scan_distance = Distance < threshold
  return scan_distance

def Comparison_Faces(img1, img2):
  image1 = reading_img(img1)
  image2 = reading_img(img2)

  face1 = Face_Detection_FaceNet(image1, margin=0.2)
  face2 = Face_Detection_FaceNet(image2, margin=0.2)

  if face1 is not None and face2 is not None:
    plt.style.use('dark_background')
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(face1)
    plt.title(f'Image 1: \nThe shape{face1.shape}')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(face2)
    plt.title(f'Image 2: \nThe shape{face2.shape}')
    plt.axis('off')

    plt.show()

    embedding1 = Embedding(face1)
    embedding2 = Embedding(face2)

    return Similarity_Measurement(embedding1, embedding2)

  else:
    return None

def Showing(image_path1, image_path2):
  print(f' {Color.BOLD}The process has been started for showing the images .....')

  checking = Comparison_Faces(img1 = image_path1, img2= image_path2)

  if checking is None:
    print(f'{Color.RED}Face not detected in one or both images. Skipping comparison.{Color.END}')
  elif checking:
    print(f'Result: {Color.GREEN} The person in both images is the {Color.BOLD}same{Color.END}')
  else:
    print(f'Result: {Color.RED} The person in both images is not the {Color.BOLD}same{Color.END}')

  print('---'*40)

  
for img in image_list:
  Showing(image_list[0], img)