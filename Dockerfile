# set base image (host OS)
FROM zaandahl/mewc-flow:1.0.11

# set the working directory in the container
WORKDIR /code

# copy the content of the local src directory to the working directory
COPY src/ .

# run training script
CMD [ "python", "./mewc_train.py" ]
