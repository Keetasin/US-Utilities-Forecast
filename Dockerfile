FROM apache/airflow:2.9.2

ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.12
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# --------------------------
# ติดตั้ง Java + Spark Runtime
# --------------------------
USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk curl procps && \
    rm -rf /var/lib/apt/lists/*

# ติดตั้ง Spark 3.5.1
RUN curl -fsSL https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz \
    | tar -xz -C /opt/ && \
    ln -s /opt/spark-3.5.1-bin-hadoop3 /opt/spark

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
USER airflow

# --------------------------
# ติดตั้ง dependencies
# --------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -c "${CONSTRAINT_URL}"
