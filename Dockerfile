FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3.5-4B', local_dir='/models/qwen')" && \
    rm -rf /root/.cache/huggingface

COPY app/app.py /app/app.py
RUN mkdir -p /app/input /app/output

ENV OCR_MODEL_PATH=/models/qwen
ENV OCR_OUTPUT_DIR=/app/output

CMD ["bash", "-c", "python3 /app/app.py --worker & exec sleep infinity"]

RUN rm -rf /root/.cache/pip /root/.cache/huggingface

RUN rm -rf /root/.cache/pip /root/.cache/huggingface
