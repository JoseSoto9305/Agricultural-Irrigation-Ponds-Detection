# Agricultural Irrigation Ponds Detection

Detects agricultural water reservoir or ponds with geomembrane lining <br>
mostly used in irrigation of crops using [google map satellite imagery](https://developers.google.com/maps/documentation/maps-static/intro) <br>
It uses a Fully Convolutional Network with ResNet50 as feature extractor. <br>
Our fine-tuned model detects irrigation ponds greater than 230 m² of size <br>
with a F1 score of 0.91 (Recall=0.90, Precision=0.92)


![f2](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/blob/master/Images/f2.png)

![f1](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/blob/master/Images/f1.png)



* Because legal restrictions of the data sources, we can not provide the original training and validation sets.
* We will provide trained model and fine-tuned parameters after the research publication of this work will be accepted.


### INSTALLATION

The codes have been tested with Python 3.6 and 3.7 versions on Debian based <br>
linux distributions (Mint 19, Ubuntu 16). It is strongly recommended to install <br>
Python libraries in a virtual environment.

To install the required libraries run:<br>
`pip install -r requirements.txt`<br>

To run the Jupyter notebooks, we recommend to install the libraries in a conda environment.<br>
`conda env create -f ./notebooks/environment.yml`<br>
`conda activate py37`<br>
`python -m ipykernel install --user --name py37 --display-name "Python (py37)"`<br>


### USAGE

See [jupyter notebooks](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/tree/master/notebooks) for a further explanation<br>

To train the model run:<br>
`python training.py`<br>

To perform predictions with the validation set run:<br>
`python predict.py`<br>

Then, to evaluate the performance of the model run:<br>
`python evaluation.py`<br>

If you want to export the predicted segmentations masks to<br>
an ESRI shapefile you can run:<br>
`python polygons_extraction.py`<br>

If you run evaluation.py or polygons_extraction.py,<br>
you need a [csv file](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/tree/master/Data/Images/Validation_Images) with the center reference coordinate for each image:<br>

center_long | center_lat | filename 
----------- | ---------- | ---------
-101.00 | 19.45 | sample_0_data.png
-101.10 | 19.55 | sample_1_data.png
-101.20 | 19.65 | sample_2_data.png
-101.20 | 19.65 | sample_n_data.png


If it is useful for your work, please cite us as:
    CITA
