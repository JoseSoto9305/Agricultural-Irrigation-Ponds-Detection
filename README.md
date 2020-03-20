# Agricultural Irrigation Ponds Detection

### Description

<br>

Detects agricultural water reservoir or ponds with geomembrane lining using [google map satellite imagery](https://developers.google.com/maps/documentation/maps-static/intro). 
It uses a Fully Convolutional Network with ResNet50 as a feature extractor. Our fine-tuned model detects 
irrigation ponds greater than 230 m² of size with an F1 score of 0.91 (Recall=0.90, Precision=0.92).

![f1](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/blob/master/Images/f1.png)

![f2](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/blob/master/Images/f2.png)


### A Case Of Study

<br>

We tested our trained model in the areas with the highest avocado production in Mexico (the green gold fringe at Michoacan 
state). The information provided by the neural network allowed the generation of a remote sensing-based inventory of water 
ponds associated with the avocado crop. See some of our results in this [jupyter notebook](https://github.com/JoseSoto9305/Agricultural-Irrigation-ponds-Detection/blob/master/notebooks/Avocado_Related_Irrigation_Ponds.ipynb).

![f3](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/blob/master/Images/f3.png)

![f4](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/blob/master/Images/f4.png)

<br> 

* Because the legal restrictions of the data sources, we can not provide the original training and validation sets.
* We will provide trained model and fine-tuned parameters after the research publication of this work will be accepted.
* Michoacán's avocado fringe is taken from: Morales, L.M.M. and G. Cuevas, G. (2012). Inventario 2011 y evaluación del
                                            impacto ambiental forestal del cultivo del aguacate en el estado de Michoacán.
                                            Informe Ejecutivo. Centro de Investigaciones en Geografía Ambiental, UNAM.
                                            Morelia, MICH. 10 pp.

---

### INSTALLATION

<br> 

The codes have been tested with Python 3.6 and 3.7 versions on Debian based linux distributions (Mint 19, Ubuntu 16).<br>
<br> 
You will need the following third-party libraries:<br>

<br>

* numpy
* pandas
* scipy
* scikit-image
* matplotlib
* seaborn
* geopandas
* shapely
* rtree
* pyproj
* tensorflow
* keras

We recommend installing dependencies in a virtual environment. To install them run:<br>
```
pip install -r requirements.txt
```

To run the Jupyter notebooks, we recommend installing the libraries in a conda environment.<br>
```
conda env create -f ./notebooks/environment.yml
conda activate py37
python -m ipykernel install --user --name py37 --display-name "Python (py37)"
```

---

### USAGE

<br> 

See [jupyter notebooks](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/tree/master/notebooks) for a further explanation:<br>
<br>

To train the model run:<br>
```
python training.py
```

To perform predictions with the validation set run:<br>
```
python predict.py
```

Then, to evaluate the performance of the model run:<br>
```
python evaluation.py
```

If you want to export the predicted segmentations masks to an ESRI shapefile you can run:<br>
```
python polygons_extraction.py
```

If you run **evaluation.py** or **polygons_extraction.py**, you need a [csv file](https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/tree/master/Data/Images/Validation_Images) with the center reference coordinate <br>
for each sample:<br>

center_long | center_lat | filename 
----------- | ---------- | ---------
-101.00 | 19.45 | sample_0_data.png
-101.10 | 19.55 | sample_1_data.png
-101.20 | 19.65 | sample_2_data.png
-101.20 | 19.75 | sample_n_data.png


---

### Reference

<br> 

If it is useful for your work, please cite us as:<br>

Soto-González, J.T. and S. Guzmán-Díaz. (2020). Agricultural Irrigation Ponds Detection. https://github.com/JoseSoto9305/Agricultural-Irrigation-Ponds-Detection/blob/master/notebooks/Avocado_Related_Irrigation_Ponds.ipynb
