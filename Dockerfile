# Start from a small, official Python base image.
# "slim" versions are much smaller than the default Python image,
# which keeps our final container lean.
FROM python:3.13-slim

# Set the working directory inside the container.
# Every command below runs relative to this folder.
WORKDIR /app

# Copy just requirements.txt first (not the whole project yet).
# This is a deliberate ordering trick: Docker caches each step, and
# dependencies change far less often than your code. By installing
# dependencies before copying the rest of the code, Docker can reuse
# this slow step from cache on future builds, as long as
# requirements.txt hasn't changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project files into the container.
COPY app.py .
COPY model.joblib .

# Document which port the app listens on (informational — doesn't
# actually publish the port by itself, see the docker run command).
EXPOSE 8000

# The command that runs when the container starts.
# 0.0.0.0 (not 127.0.0.1) is important — it means "accept connections
# from outside the container," not just from within it.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
