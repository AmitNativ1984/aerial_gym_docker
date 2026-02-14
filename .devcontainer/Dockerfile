# Devcontainer Dockerfile
# Builds on top of the pre-built aerial-gym image and adds dev dependencies

FROM localhost:5000/aerial-gym:latest

# Install dev dependencies
RUN pip install \
    loguru \
    snntorch

# Create non-root user matching host UID/GID
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && apt-get update && apt-get install -y sudo \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && rm -rf /var/lib/apt/lists/*

# Give the user access to key directories
RUN chown -R ${USER_UID}:${USER_GID} /app \
    && chown -R ${USER_UID}:${USER_GID} /opt/isaacgym

USER ${USERNAME}

# Set working directory to the aerial gym simulator
WORKDIR /app/aerial_gym/aerial_gym_simulator

CMD ["bash"]
