# Use TensorFlow GPU base image
FROM tensorflow/tensorflow:2.12.0-gpu

# Install Jupyter Notebook
RUN pip install jupyter keras matplotlib numpy

# Set the working directory
WORKDIR /tf/notebooks

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--notebook-dir=/tf/notebooks", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

