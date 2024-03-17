import os.path

import cv2
import python_package.insightface as insightface
from python_package.insightface.app import FaceAnalysis
from python_package.insightface.data import get_image as ins_get_image


if __name__ == '__main__':
    md_path = '/media/disk/business1/insightface-master'
    app = FaceAnalysis(name='buffalo_l', root=md_path)
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model('/media/disk/business1/insightface-master/models/inswapper_128.onnx')

    img1 = ins_get_image('/media/disk/business1/BeautiPho-main/models/InsightFace/t1.jpg')
    faces1 = app.get(img1)
    faces1 = sorted(faces1, key = lambda x : x.bbox[0])

    img2 = ins_get_image('/media/disk/business1/BeautiPho-main/models/InsightFace/rose1.jpeg')
    faces2 = app.get(img2)
    faces2 = sorted(faces2, key = lambda x : x.bbox[0])
    source_face = faces2[0]

    for face in faces1:
        img1 = swapper.get(img1, face, source_face, paste_back=True)

    cv2.imwrite("/media/disk/business1/BeautiPho-main/models/InsightFace/t1_swapped_rose1.jpg", img1)