import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_platform")

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    DRIFT_KS_PVALUE_THRESHOLD = float(os.getenv("DRIFT_KS_PVALUE_THRESHOLD", "0.05"))

    CANARY_TRAFFIC_PCT = int(os.getenv("CANARY_TRAFFIC_PCT", "10"))
    CANARY_SUCCESS_THRESHOLD = float(os.getenv("CANARY_SUCCESS_THRESHOLD", "0.95"))
    CANARY_DURATION_MINUTES = int(os.getenv("CANARY_DURATION_MINUTES", "30"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
