# From the Python container
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy the environment
COPY environment.yml .

# create the environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Perform package checks
RUN echo "Make sure torch is installed:"
RUN python -c "import torch"
RUN echo "Is torch installed w/ CUDA:"
RUN python -c "import torch; print (torch.cuda.is_available())"
RUN echo "Make sure tabulate is installed:"
RUN python -c "import tabulate"
RUN echo "Make sure pandas is installed:"
RUN python -c "import pandas"
RUN echo "Make sure numpy is installed:"
RUN python -c "import numpy"

# Copy the garbage
COPY src/ .
COPY assets/ .
COPY .gitignore .
COPY README.md .
COPY LICENSE .

# run main
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "crypto", "python", "./main.py"]