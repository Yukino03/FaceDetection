#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2

# In Camera
camera = 0

parser = argparse.ArgumentParser()
parser.add_argument("mask", help="Image file.(*.jpg *.png)")
parser.add_argument("cascade",nargs="*", help="Cascade Classifier. (*.xml)")
parser.add_argument("-s", "--source", help="Input file name. (Optional)")
parser.add_argument("-o", "--out", help="Output file name of video.")
#parser.add_argument("-a", "--alpha", action="store_true", default=False, help="Enable alpha blending. (for PNG file)")
args = parser.parse_args()

if args.source is not None:
  camera = args.source

if args.out is None:
  is_record = False
else:
  is_record = True

#                                      0 1 2 3 4
def detect_face(src, cascade):# return [x,y,w,h]
  img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=(170, 170))
  return facerect


def compos_rect(rects, img):
  color = (0, 255, 0)  # 矩形の色(BGR)
  if len(rects) > 0:
    for rect in rects:
      cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=6)
  return img


def compos_image(rects, img, mask):
  if len(rects) > 0:
    for rect in rects:
      mask = cv2.resize(mask, tuple(rect[2:4]))
      img[rect[1]:rect[2] + rect[1], rect[0]:rect[3] + rect[0]] = mask
  return img

def compos_image_a(rects, img, mask):#support for PNG alpha blending.FIXME
  if len(rects) > 0:
    for x, y, w, h in rects:
      mask = cv2.resize(mask, (w, h))
      img[y:y + h, x:x + w] *= 1-alpha
      img[y:y + h, x:x + w] += mask*alpha
  return img


if __name__ == "__main__":
  classifiers = []
  for casc in args.cascade:
    classifiers.append(cv2.CascadeClassifier(casc))
    mask = cv2.imread(args.mask,-1)
  history = ()
  cap = cv2.VideoCapture(camera)

  if is_record:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(args.out, fourcc, 30, (848, 480))

  while True:
    ret, img = cap.read()
    if ret == False:
      break
    #cas_num = 0
    for cascade in classifiers:# 複数の分類器を順に適用.検知したら終了.
      #cas_num = cas_num+1
      rects = detect_face(img, cascade)
      if len(rects) != 0:
        break
    #print("DEBUG:" + str(cas_num))

    if len(rects) == 0:#全ての分類器が検知しない場合.
      rects = history
    history = rects
    # dest=compos_rect(rects,im)
    dest = compos_image(rects, img, mask)
    cv2.imshow("In Camera", dest)
    if is_record:
      out.write(dest)
    key = cv2.waitKey(10)
    if key == 27:
      break
  if is_record:
    out.release()
  cap.release()
  cv2.destroyAllWindows()
