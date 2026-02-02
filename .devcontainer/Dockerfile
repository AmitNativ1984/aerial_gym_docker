# Devcontainer Dockerfile
# Builds on top of the pre-built aerial-gym image and adds dev dependencies

FROM localhost:5000/aerial-gym:latest

# Install dev dependencies
RUN pip install \
    loguru \
    snntorch

# Set working directory to the aerial gym simulator
WORKDIR /app/aerial_gym/aerial_gym_simulator

CMD ["bash"]
