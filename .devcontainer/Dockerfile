FROM osrf/ros:noetic-desktop-full

# Allow selecting ROS distro at build time (keeps default matching base image)
ARG ROS_DISTRO=noetic

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo \
    python3-dev \
    python3-pip &&\
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install common tools
RUN sudo apt install -y --no-install-recommends git wget ca-certificates lsb-release gnupg2 build-essential && \
    sudo rm -rf /var/lib/apt/lists/*

#########################################
# Install UR drivers and MoveIt packages
#########################################
# These are installed at image build time so the image contains the ROS packages referenced
RUN sudo apt-get update && \
    sudo apt-get install -y \
        ros-${ROS_DISTRO}-ur-robot-driver \
        ros-${ROS_DISTRO}-ur-calibration \
        ros-noetic-moveit-visual-tools \
        ros-noetic-moveit \
        ros-${ROS_DISTRO}-universal-robots && \
    pip install python-fcl==0.6.11 && \
    sudo rm -rf /var/lib/apt/lists/*

# Note: rosdep init/update is typically run at container runtime (postCreateCommand) because
# it needs network access and sometimes file permissions; we keep setup.bash sourced here.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

################################
## ADD ANY CUSTOM SETUP BELOW ##
################################
