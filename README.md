# Singapore Maritime Dataset Research

Welcome to the Singapore Maritime Dataset Research repository! This repository contains code and instructions for conducting research on the Singapore Maritime Dataset.

## Getting Started

To use this repository, follow these steps:

1. **Install Docker**: Ensure that you have Docker installed on your system. You can download and install Docker for your operating system from the official website: [Docker](https://www.docker.com/get-started).

2. **Clone the Repository**: Clone this repository to your local machine using the following command:
   
   ```bash
   git clone https://github.com/rmmbdev/object-detection.git
   ```
3. **Download Singapore Maritime Dataset**: Download the Singapore Maritime Dataset from the following Kaggle link and extract it:

   https://www.kaggle.com/datasets/mmichelli/singapore-maritime-dataset

4. **Build the Docker Image**: Build the Docker image for this project by running the following command in the project's root directory:
   ```bash
   docker build --tag=obj-04 . 
   ```
5. **Update docker-compose.yml**: Update the `docker-compose.yml` file with the appropriate information related to the extracted dataset archive file.
6. **Run the Experiment**: Start the experiment by running the following command:
   ```bash
   docker-compose up
   ```