import cv2
import yaml

fs = cv2.FileStorage("lr_weights.yaml", cv2.FILE_STORAGE_READ)
fn = fs.getNode("coef")
ym = fn.mat().tolist()
print(yaml.dump(ym))