## Recognize Anything ##

### **Setting Up** ###

```
cd recognize-anything
pip install setuptools --upgrade
pip install -e .
```

### Checkpoint ###
put the following model to pretrained folder:

https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth?download=true

## Grounded SAM ##

### **Setting Up** ###

```
cd Grounded-Segment-Anything
```

Follow the instruction in README.md

## Inference ##

```
cd recognize-anything
python3 demo_final.py --image images/demo/demo1.jpg
```