# Agricultural Irrigation Ponds Detection


Detects agricultural water reservoir or ponds with geomembrane lining mostly <br>
used in irrigation of crops <br> using [google map satellite imagery!](https://developers.google.com/maps/documentation/maps-static/intro) <br>
It uses a Fully Convolutional Network with ResNet50 as feature extractor. <br>
Our fine-tuned model detects irrigation ponds greater than 230 m² of size <br>
with a F1 score of 0.91 (Recall=0.90, Precision=0.92)


Imagen

Image description: From left to right: RGB Image from Google Map Satellite; 
Ground truth binary segmentation; output, probability of belonging to the category, 0 is no probability, 1 is full probability.


*** Because legal restrictions of the data sources, we can not provide the original training and validation sets.

**  We will provide trained model and fine-tuned parameters after the research publication of this work will be accepted.


INSTALLATION

The codes have been tested with Python 3.6 and 3.7 versions on Debian based linux distributions (Mint 19, Ubuntu 16).
It is strongly recommended to install Python libraries in a virtual environment.


To install the required libraries run:
`<pip install -r requirements.txt>`

To run the Jupyter notebooks, we recommend to install the libraries in a conda environment.

conda env create -f ./notebooks/environment.yml 
conda activate py37
python -m ipykernel install --user --name py37 --display-name "Python (py37)"


USAGE

See jupyter notebooks for a further explanation (link)

The codes assume the data is stored in this format:

. src
    . Data
        . Images
            . Train_Images
                . sample_0
                    sample_0_data.png
                    sample_0_labels.png
                . sample_1
                    sample_1_data.png
                    sample_1_labels.png
                . sample_2
                    sample_2_data.png
                    sample_2_labels.png            
                ...
                ...

            . Validation_Images
                . sample_0
                    sample_0_data.png
                    sample_0_labels.png
                . sample_1
                    sample_1_data.png
                    sample_1_labels.png
                . sample_2
                    sample_2_data.png
                    sample_2_labels.png

                ...
                ...


To train (calibrate) the model run
python training.py

To perform predictions with the validation set run:
python predict.py

Then, to evaluate the performance of the model run:
python evaluation.py

If you want to export the predicted segmentations masks to 
an ESRI shapefile you can run:

python polygons_extraction.py

If you run evaluation.py or polygons_extraction.py, 
you need a csv file with the center reference coordinate for each image:


| center_long |  center_lat |    filename |
  -101.00          19.45       sample_0_data.png
  -101.10           19.55       sample_1_data.png
  -101.20           19.65       sample_2_data.png
  ...              ...         ...
  ...              ...         ...


If it is useful for your work, please cite us as:
    CITA
