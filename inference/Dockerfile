FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y unzip

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install -r requirements.txt


RUN pip install gdown

RUN gdown https://drive.google.com/uc?id=1fUBAkrXySeGDahfdWziG4Is2loJ9Dom0 



# Copy the rest of the code
COPY . .


# Expose the port used by Streamlit
EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
