FROM apache/airflow:2.9.2-python3.12

ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.12
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk curl procps && rm -rf /var/lib/apt/lists/*

ENV SPARK_VERSION=3.5.1
ENV SPARK_ARCHIVE_URL=https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz

RUN curl -fSL "${SPARK_ARCHIVE_URL}" -o /tmp/spark.tgz \
    && tar -xzf /tmp/spark.tgz -C /opt/ \
    && ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop3 /opt/spark \
    && rm /tmp/spark.tgz

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH="$PATH:${SPARK_HOME}/bin:${SPARK_HOME}/sbin"

USER airflow

COPY --chown=airflow:root requirements.txt /opt/airflow/requirements.txt

RUN awk '/^apache-airflow-providers-/{print $0}' /opt/airflow/requirements.txt > /opt/airflow/req.providers.txt && \
    if [ -s /opt/airflow/req.providers.txt ]; then \
      pip install --no-cache-dir -r /opt/airflow/req.providers.txt -c "${CONSTRAINT_URL}"; \
    fi

RUN awk '!/^apache-airflow-providers-/' /opt/airflow/requirements.txt > /opt/airflow/req.noprov.txt && \
    pip install --no-cache-dir -r /opt/airflow/req.noprov.txt
