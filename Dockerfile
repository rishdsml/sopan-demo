# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files from local folder to container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 7860

# Run your app
CMD ["streamlit", "run", "ncr_chatbot.py", "--server.port=7860", "--server.enableCORS=false"]
