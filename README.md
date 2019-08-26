# PIC_HOIW
This repository provides python API that assists in loading, parsing and visualizing data in HOIW.
## Install
1. Clone this repo:

    ~~~
    git clone https://github.com/YueLiao/PIC_HOIW.git
    ~~~


2. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~

## Use API
For HOI visulization on an image, run:

~~~
python demo.py --image_dir ./iccv_hoiw/images/trainval --annot_file ./iccv_hoiw/annot/trainval.json --image_name trainval_000001.png
~~~
For HOI visulization on all images in an annotation file, run:

~~~
python demo.py --image_dir ./iccv_hoiw/images/trainval --annot_file ./iccv_hoiw/annot/trainval.json --output_path vis_hoiw
~~~


