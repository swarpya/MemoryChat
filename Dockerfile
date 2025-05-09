# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Use --no-cache-dir to avoid storing cache in the image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (including templates and static)
COPY . .

# Expose the port the app runs on (Hugging Face requires 7860 for web apps)
EXPOSE 7860

# Define the command to run your application using Waitress
# Waitress serves the 'app' object from your 'app.py' file
# The --listen 0.0.0.0:7860 ensures it listens on the correct address and port
CMD ["waitress-serve", "--listen=0.0.0.0:7860", "app:app"]