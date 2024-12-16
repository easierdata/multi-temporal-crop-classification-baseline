# Baseline Model for Segmentation Fine-Tuning of the HLS Foundation Model

This repo, a revision originally [based](https://github.com/ClarkCGA/multi-temporal-crop-classification-training-data) from [Clark Center for Geospatial Analytics](https://www.clarku.edu/centers/geospatial-analytics/), to run a supervised CNN model pipeline for multi-temporal crop type segmentation, based on the HLS Foundation Model (FM). The FM is released by NASA and IBM [here](https://huggingface.co/ibm-nasa-geospatial), and the fine-tuned FM model for this task is presented [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification).

The pipeline includes, model training, evaluation, and inference, for data generated in the [multi-temporal-crop-classification-training-data](https://github.com/easierdata/multi-temporal-crop-classification-training-data).

## Prerequisites

To get started:

1. change directory to an empty folder in your machine and clone the repo.

```
$ cd /to_empty/dir/on_host/

$ git clone https://github.com/easierdata/multi-temporal-crop-classification-baseline

$ cd path/to/cloned directory/
```

2. create a virtual environment and install the required dependencies.

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Model Pipeline

### Configuring pipeline parameters

A configuration file, found [here](./config/default_config.yaml), is used to customize various model properties and features for model training, evaluation, and inference. The `custom dataset params` section in the configuration file is used to specify the dataset path, batch size, and other dataset-related parameters. Supply a `dataset_path` containing your dataset content and the `dataset_dir` containing the images to be used for training, validation, or inference.

> :white_check_mark: It's important to note that the data sent through the pipeline must be preprocessed. Two files are expected for each chip:
> 1. Three HLS scenes merged together, with a naming convention of `chip_xxx_xxx_merged.tif`
> 2. Label chip contaning reclassified CDL crop types (13 classes total), with a naming convention of `chip_xxx_xxx.mask.tif:`
> 
> More details on the preprocessing can be found [here](https://github.com/easierdata/multi-temporal-crop-classification-training-data/doc/Training%20Data%20Overview.md#import-saved-dataframe-files-and-preparing-tile-chipping-process).

### Running the pipeline in different modes

Run the pipeline by executing the following command and passing in one of the following modes: `train`, `validation`, or `inference`.

```shell
python ./src/run_model.py --config_file <path_to_config_file> <mode>
```

> By default, the configuration file is set to `config/default_config.yaml`. You can specify a different configuration file by providing the path to the file using the `--config_file` argument.

The results of the pipeline will be saved to a directory, as specified by the `working_dir` parameter in the configuration file. The results include:

##### **Predictions**

Running the pipeline in `inference` mode will generate and save the predictions to the `working_dir` directory.

##### **Model Checkpoints**

The model checkpoints are saved to the `output_dir` directory. Model checkpoints represent the number of epochs the model has been trained on and are saved based on the `checkpoint` parameter in the configuration file. Files are saved in the format `model_epoch_<epoch_number>.pth.tar`.

##### **Model Params**

The model parameters are saved to the `output_dir` directory after a model has finished training. Running the `save` in the [ModelCompiler Class](./src/model_compiler.py/) The model parameters are saved in the format `model_params_epoch_<epoch_number>.pth`. The model parameters can be loaded and used for inference or warm-up training by passing in the file path to the `params_init` parameter in the configuration file.

##### **Evaluation metrics**

Example below of the metrics generated from the pipeline:

![Confusion Matrix](_media/confusion_matrix.png)

and two CSV files containing the evaluation metrics and the model predictions.

##### **Overall Metrics**

|Metric          |Value   |
|----------------|--------|
|Overall Accuracy|0.63056 |
|Mean Accuracy   |0.61915 |
|Mean IoU        |0.42086 |
|mean Precision  |0.57392 |
|mean Recall     |0.57492 |
|Mean F1 Score   |0.57251 |

##### **Class-wise Metrics**

|Class               | Accuracy   |IoU         |Precision  |Recall       |F1 Score    |
|--------------------|------------|------------|-----------|-------------|------------|
|Natural Vegetation  |0.6366      |0.4577      |0.6196     |0.6366       |0.6280      |
|Forest              |0.7171      |0.4772      |0.5878     |0.7171       |0.6461      |
|Corn                |0.6332      |0.5226      |0.7494     |0.6332       |0.6864      |
|Soybeans            |0.6676      |0.51675     |0.6957     |0.6676       |0.6814      |
|Wetlands            |0.6035      |0.4109      |0.5628     |0.6035       |0.5825      |
|Developed/Barren    |0.6022      |0.4637      |0.6684     |0.6022       |0.6336      |
|Open Water          |0.8775      |0.7596      |0.8496     |0.8775       |0.8633      |
|Winter Wheat        |0.6639      |0.4950      |0.6606     |0.6639       |0.6622      |
|Alfalfa             |0.5902      |0.3847      |0.5250     |0.5902       |0.5557      |
|Fallow/Idle Cropland|0.5293      |0.3599      |0.5292     |0.5293       |0.5293      |
|Cotton              |0.4529      |0.3258      |0.5371     |0.4529       |0.4914      |
|Sorghum             |0.6152      |0.3909      |0.5174     |0.6152       |0.5621      |
|Other               |0.4589      |0.3268      |0.5316     |0.4589       |0.4926      |

### Running the pipeline from Jupyter Notebook

You can also run the pipeline from a Jupyter Notebook to walk through the different steps in running a model. 

Execute the following command:

```shell
jupyter notebook
```

Then navigate to the `notebooks` directory and open the `main.ipynb` notebook. Follow the instructions in the notebook to run the pipeline

## Instructions to run the code using Docker

**Step 1** Make sure the Docker daemon is running and build the Docker image as following:

``` bash
$ docker build -t <image_name>:<tag> .
```

Example:

``` bash
$ docker build -t semseg_baseline:v1 .
```

**step 2-** Run the Docker image as a container from within the cloned folder:

``` bash
$ docker run --gpus all -it -p 8888:8888 -v <path/to/the/cloned-repo/on-host>:/home/workdir -v <path/to/the/dataset/on-host>:/home/data  <image_name>:<tag>
```

This command will start a container based on the specified Docker image and starts a JupyterLab session. Type `localhost:8888` in your browser and copy the provided token from the terminal to open the JupyterLab.

**step 3-** Run the pipeline:

Modify the "default_config.yaml" or create your own config file and run the cells as explained in the notebook.

## Model Weights

The model weights trained on the dataset for 100 epochs with the parameters specified in the "default_config.yaml", is stored in the `model_weights/multi_temporal_crop_classification.pth`. Instructions to load and use the pre-trained model for zero-shot inference or warm-up training is explained in the notebook.
