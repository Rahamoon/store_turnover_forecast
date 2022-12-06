# Store turnover forecast
A Machine learning pipeline for training and predicting sales turnover

Table of Contents
-----------------

  * [Requirements](#requirements)
  * [Usage](#usage)



# Requirements
The Pipeline requires the following to run:
+ docker

# Usage
Clone the repo:
```sh
git clone git@github.com:Rahamoon/store_turnover_forecast.git
```

Build the docker image:
```sh
docker build -t {image_tag} {project_path}
```

Run the docker container for training:
```sh
docker run -v {model_local_path}:/train {image_tag} --mode train --model_path /train/model
```

Run the docker container for predicting:
```sh
docker run -v {folder_path}:/serve turnover_forecast --mode serve --model_path /serve/model  --results_path /serve/data
```