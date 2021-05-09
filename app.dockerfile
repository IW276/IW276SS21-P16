FROM  docker.pkg.github.com/iw276/iw276ss21-p16/base_image:0.2
COPY src/mean.npy /app
COPY src/std.npy /app
COPY src/applyModel.py /app
COPY pretrained-models/mediocreModel_0.pt /app

ENTRYPOINT ["python3", "applyModel.py"]

