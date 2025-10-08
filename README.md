# US Utilities Sector Stock Forecasting

## Description
This project is an all-in-one web application for stock analysis and forecasting. It features an interactive dashboard that allows users to view stock details, historical price trends, key financial indicators, and more in real time. The system supports multiple forecasting models, including ARIMA, SARIMA, SARIMAX, and LSTM, along with a backtesting feature that evaluates model accuracy using MAE (Mean Absolute Error) as the performance metric.

The application can forecast stock prices over 1 week, 6 months, and 1 year, and enables model comparison to identify the best-performing approach. It also provides the 5 latest relevant news articles. Integrated with Airflow, the system trains models and automatically updates data from Monday to Friday, storing all information in PostgreSQL. This platform empowers users to analyze trends, assess model accuracy, and make strategic, data-driven decisions effectively.

## Installation
Ensure that you have Python 3.9+, Docker, and Docker Compose installed on your system before proceeding.

1. Clone the repository
   ```bash
   git clone https://github.com/Keetasin/US-Utilities-Forecast.git
   ```
2. Navigate to the project directory
   ```bash
   cd US-Utilities-Forecast
   ```
3. Add your NewsAPI key
   - Sign up and get your API key from https://newsapi.org/
   - Open the file src/web/utils/news.py
   - Replace API_KEY with your actual key:
   ```bash
   NEWS_API_KEY = "API_KEY" 
   ```
   
4. Build the Docker image
   ```bash
   docker build -t custom-airflow:latest .
   ```

## Usage
1. Start all services
   ```bash
    docker-compose up -d
   ```
2. Access the Airflow web UI 
   ```
   http://localhost:8080
   ```
   - **Username:** admin
   - **Password:** admin 
   - After logging in:
        1. Go to **Admin > Connections**.
        2. Find and edit the record **spark_default**.
        3. Set **Host** to `local[*]`.
        4. Click **Save**.
   - Unpause DAG

3. Access the Flask web application
   - Wait for Airflow to finish running (especially on the first run), then open
   ```
   http://localhost:5000
   ```
4. Access pgAdmin 4 (local application)
   - Open pgAdmin 4 on your local machine.
   - Create a new server with the following settings:
     - **Name:** US-Utilities-DB (or any name you like)
     - **Host name/address:** localhost
     - **Port:** 5432
     - **Username:** airflow
     - **Password:** airflow
   - Save and connect. 

## Services
| Service        | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `postgres`     | PostgreSQL database for Airflow metadata and application data. |
| `airflow-init` | Initializes Airflow DB and creates an admin user.              |
| `webserver`    | Airflow web UI accessible at `http://localhost:8080`.          |
| `scheduler`    | Airflow scheduler that runs DAGs.                              |
| `web`          | Flask web application accessible at `http://localhost:5000`.   |

## Authors 
- 6610110214 Peeranat Pathomkul
- 6610110425 Keetasin Kongsee
- 6610110475 Natdanai Chookool









