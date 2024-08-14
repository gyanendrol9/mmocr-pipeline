# mmocr-pipeline

## Description

This is an API to assist in processing and creating configurations and JSON files needed to
train and test on the MMOCR library. As part of this README file, I have included general
environment setup for running this pipeline as well as the setup for running MMOCR.

## Features
- All you need is a single file for all tasks
- Generate Configuration and JSON files for multiple datasets
- Generate basic training/testing configuration file for available detection and recognition frameworks in MMOCR
- Includes a Box Transformer class to transform between different types of bounding boxes (currently supports types: VOC, ICDAR, QUAD, COCO)
- Includes a custom generator to generate bounding boxes given text and image using MMOCR (NOT FINISHED YET)
- Can include custom dataset by defining a class for each under dataset_lib folder (all you need to define is the process function, by inheriting from MMOCRDataset)



## Environment Configuration

### Create Conda Environment

Preferred Python version 3.8, limit 3.10
```commandline
>>> cd <<Directory of Choice>>
>>> conda create --name ocrenv
>>> conda activate ocrenv
>>> conda install python=3.8 conda pip 
```

### Install Dependencies

#### Install Pytorch (CUDA: 11.7 is preferred) following Link (https://pytorch.org/get-started/locally/)
```commandline
>>> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### Setup MMOCR
```commandline
>>> pip install -U openmim
>>> mim install mmengine
>>> mim install ‘mmcv>=2.0.rc1’
>>> mim install ‘mmdet==3.0.0rc5’ 
>>> git clone https://github.com/open-mmlab/mmocr.git 
>>> cd mmocr 
>>> pip install -r requirements.txt 
>> pip install -v -e .
```

#### Setup Pipeline

Recommend cloning under mmocr directory or in the same directory as mmocr for ease use (little navigation)
```commandline
>>> git clone https://github.com/ipiriyan2002/mmocr-pipeline.git
```

#### Others (Make sure to check if the following has already been installed alongside the former)
```commandline
>>> conda install pillow
>>> conda install pandas
>>> conda install numpy
>>> conda install -c conda-forge datasets 
>>> pip install levenshtein
>>> pip install pyarrow
>>> conda install -c jmcmurray json
>>> conda install -c conda-forge argparse
>>> conda install -c conda-forge albumentations
>>> conda install jsonlines
```

## Pipeline

### Setting up the single config file

Three important defenition: dataset_dict, det_model_dict and recog_model_dict

For No task (Empty configuration)
```python
dataset_dict = None

det_model_dict = None

recog_model_dict = None
```

### Dataset dict
When dataset_dict is None, no dataset's would be processed

You can either provide a list of datasets to process under dataset_dict or a single dataset

Each dataset would be defined using the following setting

```python
dataset_dict = [
    dict(
        type="cordv2",   #Available types include "cordv2", "ing", "meme", "glosat"
        config_save_dir=None, #Directory at which MMOCR dataset config are created, keep at None unless needed
        init_params= dict(name="cordv2", tasks = ["det"], save_dir=None, use_gen=True, generator=None),
        prepare_params=dict(multi=dict(img_paths=[], ann_paths=[], split=dict(train=0.8, test=0.1, val=0.1)),
                            train=dict(img_paths=None, ann_paths=None, split="train"),
                            test=dict(img_paths=None, ann_paths=None, split="test"),
                            val=dict(img_paths=None, ann_paths=None, split="val"))
    ),
]
```

The above version is a list, you can also provide just the dictionary if you want to process only that dataset

The init_params and prepare_params are common for all datasets and should not be changed

If changed in custom definition, do remember to include the parameter where needed

Currently we support the following datasets:
- CORD dataset: "cordv2"
- Ingredient dataset (custom): "ing"
- Meme dataset (custom): "meme"
- Glosat dataset: "glosat"

Supports tasks: "det" and "recog"

```
To prepare for a new dataset, place the data preparation code in the mmocr-pipeline/dataset_lib folder.
```

#### prepare params
If multi is defined and not None, the given images and annotations are used alongside split percentages to get training, testing and validation datasets.
If not, then the rest of the params are processed.

The img_paths and ann_paths accept a single (folder/file) path or a list of (folder/file) paths.

When defining custom datasets do remember to support functionality for reading files given a folder

### Defining a custom generator under init_params:

If using generator set, use_gen to True and if not set to False, to avoid downloading pre-defined weights for MMOCR models

Define the detection, recognition, custom weights, device and save directory for 
ocr-based bounding box generator.

max_neighbours (must be a odd number) defines the window size for considering neighbouring predicted text

Recommends 5 for max_neighbours

Given max_neighbours look (k-1)/2 to the right and left
```python
generator = dict(
    det_model="dbnet", det_weights=None, rec_model="satrn", rec_weights=None,
    device="cuda:0", save_dir="./box_gen/", max_neighbours=5
)
```

### Model dict

For models, we support detection and recognition configuration, where the definition for both are similar.
Like dataset_dict, you can also provide a list of dicts or a single one if needed

For the following two dicts, it is important to state has_val if providing validation file and to state False for has_val if not.

### Det Model Dict

Available schedules, backbones and necks for each model can be found under "./mmocr_config_writers/configs/det_configs.py"

```python
det_model_dict = dict(
    train_datasets=["cordv2.py"], #Single or Multiple training datasets
    val_datasets=["cordv2.py"], #Single, Multiple or None validation datasets
    test_datasets=["cordv2.py"], #Single or Multiple testing datasets
    model="dbnet", #Model name depending on task
    backbone="resnet18", #Backbone for model
    neck="fpnc", #Neck of model
    base=None, #Base not needed to be defined for detection model
    vocab=None, #Vocabulary for recognition models
    epochs=40, #Maximum epoch
    schedule=None, #Predefined Learning rate schedulers, optimizers
    has_val=True, # If validation is included
    train_batch_size=16, # Batch size for training
    test_batch_size=1, # Batch size for testing and validation
    contents = dict( #Dict defining the hyperparameters
        log_interval=1, # Log at each k epoch
        checkpoint_interval=1, # Save at each k epoch
        optimizer_params=dict( #Follows pytorch defenitions for parameters
            type="SGD", lr=0.007, momentum=0.9, weight_decay=0.0001
        ),
        schedulers=[ #List of schedulers, follows pytorch defenitions
            dict(type="ConstantLR", factor=1.0)
        ],
        cfgs=dict( #No need for val/test cfg to be defined as they are pretty 
            # much not need for changing
            #train_cfg, define max_epoch and validation interval (every k epochs, validate model)
            train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=40, val_interval=1),
            val_cfg=None,
            test_cfg=None
        ),
    )
```

### Recog Model Dict

Available schedules, bases for each model can be found under "./mmocr_config_writers/configs/recog_configs.py"

```python
recog_model_dict = dict(
    train_datasets=["cordv2.py"], #Single or Multiple training datasets
    val_datasets=["cordv2.py"], #Single, Multiple or None validation datasets
    test_datasets=["cordv2.py"], #Single or Multiple testing datasets
    model="abinet", #Model name depending on task
    backbone=None, #Backbone not needed to be defined for recognition model
    neck=None, #Neck not needed to be defined for recognition model
    base="_base_abinet-vision.py", #Base for model
    vocab=None, #Vocabulary for recognition models 
    epochs=40, #Maximum epoch
    schedule=None, #Predefined Learning rate schedulers, optimizers
    has_val=True, # If validation is included
    train_batch_size=16, # Batch size for training
    test_batch_size=1, # Batch size for testing and validation
    contents = dict( #Dict defining the hyperparameters
        log_interval=1, # Log at each k epoch
        checkpoint_interval=1, # Save at each k epoch
        optimizer_params=dict( #Follows pytorch defenitions for parameters
            type="SGD", lr=0.007, momentum=0.9, weight_decay=0.0001
        ),
        schedulers=[ #List of schedulers, follows pytorch defenitions
            dict(type="ConstantLR", factor=1.0)
        ],
        cfgs=dict( #No need for val/test cfg to be defined as they are pretty 
            # much not need for changing
            #train_cfg, define max_epoch and validation interval (every k epochs, validate model)
            train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=40, val_interval=1),
            val_cfg=None,
            test_cfg=None
        ),
    )
```

vocab can be either a list of characters or path to text document with the characters to train.

### Running prepare
```commandline
>>> cd mmocr-pipeline
```
Once above configuration file is defined run the following code with following optional flags
- '-nd' or '--no-dataset' -> Do not process defined datasets
- '-ndm' or '--no-det-model' -> Do not process defined detection model
- '-nrm' or '--no-recog-model' -> Do not process defined recognition model

```commandline
>>> python prepare.py <<config_path>>
```

Example, if custom config file is named prepare_config_2.py
```commandline
>>> python prepare.py ./prepare_config_2.py
```

### Training, Testing and Infering under MMOCR

Make sure you are in the mmocr directory and not mmocr-pipeline directory

If in mmocr-pipeline directory:
```commandline
>>> cd <<MMOCR directory>>
```

#### Training

The following commands are taken from https://mmocr.readthedocs.io/en/dev-1.x/user_guides/train_test.html
```commandline
# Train the specified MMOCR model by calling tools/train.py
>>> CUDA_VISIBLE_DEVICES= python tools/train.py ${CONFIG_FILE} [PY_ARGS]

# Training
# Example 1: Training DBNet with CPU
>>> CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py

# Example 2: Specify to train DBNet with gpu:0, specify the working directory as dbnet/, and turn on mixed precision (amp) training
>>> CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py --work-dir dbnet/ --amp
```

The generated configuration path is outputted from prepare.py. By default, it will be under the following directories:
- configs/textdet/<<model_name>>/<<config_file>>
- configs/textrecog/<<model_name>>/<<config_file>>

where the config file is named after model, backbone, neck and epochs for detection
and model, epochs for recognition.

Names of generated files are given as output from prepare.py/


#### Testing

The following commands are taken from https://mmocr.readthedocs.io/en/dev-1.x/user_guides/train_test.html

```commandline
# Test a pretrained MMOCR model by calling tools/test.py
>>> CUDA_VISIBLE_DEVICES= python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]

# Test
# Example 1: Testing DBNet with CPU
>>> CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth

# Example 2: Testing DBNet on gpu:0
>>> CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth
```

Checkpoint file can be found under work_dirs in mmocr directory, for each specific trained config file


#### Multiple GPUS

The following commands are taken from https://mmocr.readthedocs.io/en/dev-1.x/user_guides/train_test.html


```commandline
# Training
>>> NNODES=${NNODES} NODE_RANK=${NODE_RANK} PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]

# Testing
>>> NNODES=${NNODES} NODE_RANK=${NODE_RANK} PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```


#### Infering

More can be found here: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html

--show flag does not work in servers without GUI or when using SSH tunnel. 

```commandline
>>> python tools/infer.py <<Image path / Image Directory path>>\
--det <<Model name / config file under work_dirs>> or None\ 
--det-weights <<trained weights under work_dirs>> or None\
 --rec <<Model name / config file under work_dirs>> or None\
  --rec-weights <<trained weights under work_dirs>> or None\
   --show
```

If that is the case:

```commandline
>>> python tools/infer.py <<Image path / Image Directory path>>\
 --det <<Model name / config file under work_dirs>>\
  --rec <<Model name / config file under work_dirs>>\
   --out-dir <<Directory to store results>>\
   --save_vis --save_pred 
```

--save-vis : saves the visualization of inference results
--save-pred : saves the raw inference results

