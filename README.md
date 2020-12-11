## FYS-STK3155/4155 Project 3

### Set-up

One needs to have following packages installed: `tensorflow`, `keras`, `wandb`. To install those run `pip install [nameofthepackage]`. 
In addition, if one wants to use wandb, one needs to have a wandb account and log in by running `wandb login [apiloginnumber]`. See project runs on wandb [here](https://wandb.ai/fys-stk-gang/project3). But it is possible to run training without logging in.

### Running
Once everything is installed and set-up correctly. To run the training:

```
python -w ignore 'train.py'
```
There you will be prompted if you have logged in to wandb or not, if not, wandb will be disabled. Then you can choose to train one of the 3 pre-configurated models.

### Report folder
Contains Latex code and pdf of the report

### cnn folder
The main new library, see how it works together in the diagram:<br/>
![](project3.png)

### DataFiles folder
Contains the lego dataset downloaded from [Kaggle Lego](https://www.kaggle.com/joosthazelzet/lego-brick-images) and programatically separated into train, validation and test datasets.

### sweep_configs
Contains .yaml configuration files used to run tests on parameters for different neural networks.
