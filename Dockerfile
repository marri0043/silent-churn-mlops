# Step 1: Start from Python base image
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy requirements first
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all project files
COPY . .

# Step 6: Expose port
EXPOSE 8000

# Step 7: Run the API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]