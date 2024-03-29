
# Dependencies installation

## Install requirements.txt
```
pip install -r requirements.txt
```

## GroundingDINO
In the root directory
```
pip install --no-build-isolation -e GroundingDINO
```

## InsightFace: 2D and 3D Face Analysis Project


### Pretrained model downloading

1. create model folder
```
mkdir model_weights/insightface
```

2. download model to the folder
```
https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view
```

### Install mesh_core_cython package

```
cd models/InsightFace/python_package/insightface/thirdparty/face3d/mesh/cython

pip install .
```
