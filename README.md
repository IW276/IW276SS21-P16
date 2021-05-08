# Project-Template for IW276 Autonome Systeme Labor

Short introduction to project assigment.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Autor 1, Autor2, Autor 3 during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-model)
* [Running](#running)
* [Acknowledgments](#acknowledgments)

## Requirements
* Python 3.6 (or above)
* OpenCV 4.1 (or above)
* Jetson Nano
* Jetpack 4.4
> [Optional] ...

## Prerequisites
1. Install requirements:
```
pip install -r requirements.txt
```

## Pre-trained models <a name="pre-trained-models"/>

Pre-trained model is available at pretrained-models/

## Running

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
```
python src/demo.py --model model/student-jetson-model.pth --video 0
```
> Additional comment about the demo.

## Docker
We are using a base image, which includes only the requirements.
To build that image you can execute use `docker build -t docker.pkg.github.com/iw276/iw276ss21-p16/base_image -f base_image.dockerfile .`
You can download the already crated container too currently you need the read packages permission do do this.
If the base image can be found on your local system, you can build the app container with the following command `docker build -t docker.pkg.github.com/iw276/iw276ss21-p16/app -f app.dockerfile .`

If you want to start a container, you can use this command:
```
docker run --runtime=nvidia -v soundtrack:/mnt/soundtrack -v out:/mnt/out docker.pkg.github.com/iw276/iw276ss21-p16/app:latest
```

We are assume, that the tag latest exsists and references to the latest builded container image.


## Acknowledgments

This repo is based on
  - [Envirnmental-Sound-Classification](https://github.com/mariostrbac/environmental-sound-classification)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
