import os
import cv2
import csv
from joblib import Parallel, delayed
from time import time
import sys
import pickle

src_path = sys.argv[1]
classifier_path = sys.argv[2]
write_path = sys.argv[3]
#src_path = "/Users/myazdaniUCSD/Documents/selfiecity/images/selfies/thumb/"
#classifier_path = '/Applications/OpenCV/opencv-2.4.8/data/haarcascades/haarcascade_frontalface_alt.xml'
#write_path = "./results"

image_paths_list = []  
for root, dirs, files in os.walk(src_path):
  temp = [os.path.join(root, f) for f in files if f.endswith('.jpg')]
  if len(temp) > 0: image_paths_list.extend(temp)

def chunks(l, n):
  for i in xrange(0, len(l), n): yield l[i:i+n]

image_paths_list = list(chunks(image_paths_list, len(image_paths_list)/1))

face_cascade = cv2.CascadeClassifier(classifier_path)
#face_cascade = cv2.CascadeClassifier()

def return_face_dict(path):
  dict = {}
  img = cv2.imread(path)
  if img is None:
    dict['file_path'] = path
    dict['num_faces'] = 'NA'
  else:
    img = cv2.resize(img,(150,150))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.03, 5)

    dict['file_path'] = path
    dict['num_faces'] = len(faces)
    if len(faces) > 1:
      rgb_hist = []
      #rois = []
      for face in faces:
        scale = .1
        initial_x = face[1] + int(scale*face[1])
        final_x = face[1]+face[3] - int(scale*(face[1]+face[3]))
        initial_y = face[0] + int(scale*face[0])
        final_y = face[0]+face[2] - int(scale*(face[0]+face[2]))
        #roi = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2],]
        roi = img[initial_x:final_x, initial_y:final_y,]
        #rois.append(roi)
        rgb_hist.append([cv2.calcHist([roi],[i],None,[256],[0,256]) for i in range(3)])
      dict['rgb_hist'] = rgb_hist
    #dict['rois'] = rois
  return dict


t0 = time()

for i, image_paths in enumerate(image_paths_list):
  print "working on batch", str(i+1), "out of", str(len(image_paths_list))
  results= Parallel(n_jobs=-1)(delayed(return_face_dict)(image_path) for image_path in image_paths)
  #with open(write_path + '/results_dict_' + str(i) + '.pkl', 'wb') as output: pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

  output = [[result['file_path'], result['num_faces']] for result in results]

  with open(write_path + '/filepaths_faces_' +str(i) + '.csv', 'wb') as f:
    writer = csv.writer(f, delimiter = ",")
    writer.writerow(['file_path', 'num_faces'])
    for item in output: writer.writerow([item[0], item[1]])
  
print time() - t0
