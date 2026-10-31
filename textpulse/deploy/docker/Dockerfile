# Recipe Hugging Face follows to build and run TextPulse.

# Start from a small Linux image with Python 3.13 (the version the models were trained with)
FROM python:3.13-slim

# Hugging Face runs apps as a normal user with id 1000, not as administrator
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install the libraries first, so Docker can reuse this step when only code changes
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app, the code and the trained models
COPY --chown=user . .

# Hugging Face Spaces send visitors to port 7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
