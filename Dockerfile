# VERSION 0.0.21
FROM apache/airflow:2.5.3-python3.10
LABEL maintainer=ajjunior

ENV AIRFLOW_HOME=/usr/local/airflow
ENV SPARK_VERSION=3.2.1
ENV SPARK_HOME=/usr/local/spark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:/usr/local/spark/bin
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install system packages as root
USER root
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        wget git curl iputils-ping openjdk-11-jdk build-essential \
        libsasl2-dev python3-dev libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements (ยังไม่ติดตั้ง)
COPY requirements.txt /requirements.txt

# Upgrade pip/setuptools/wheel as airflow user
USER airflow
RUN python3 -m pip install --upgrade "pip<23" "setuptools<67" "wheel<0.45" --user

# Install Spark (without Hadoop) และแก้ปัญหา permission
USER root
RUN cd /tmp \
    && wget --no-verbose "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-without-hadoop.tgz" \
    && tar -xvzf "spark-${SPARK_VERSION}-bin-without-hadoop.tgz" \
    && mkdir -p "${SPARK_HOME}" \
    && cp -a "spark-${SPARK_VERSION}-bin-without-hadoop/." "${SPARK_HOME}/" \
    && rm "spark-${SPARK_VERSION}-bin-without-hadoop.tgz" \
    && chown -R airflow: ${SPARK_HOME}

# Copy entrypoint + config
COPY src/script/entrypoint.sh /entrypoint.sh
COPY src/config/airflow.cfg ${AIRFLOW_HOME}/airflow.cfg
RUN chmod +x /entrypoint.sh \
    && chown -R airflow: ${AIRFLOW_HOME}

WORKDIR ${AIRFLOW_HOME}
USER airflow
ENTRYPOINT ["/entrypoint.sh"]
CMD ["webserver"]
EXPOSE 8080 5555 8793
