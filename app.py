import os
import matplotlib
matplotlib.use('Agg')
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from functools import wraps

# Flask App Config
app = Flask(__name__)
CORS(app)
os.makedirs("static", exist_ok=True)

# Configuration
MAX_EDA_TIME = 30  # seconds
MAX_ROWS_FOR_EDA = 1000  # Limit rows for EDA

# MongoDB Connection
MONGO_URI = os.environ.get('MONGO_URI', "mongodb+srv://vgugan16:gugan2004@cluster0.qyh1fuo.mongodb.net/dL?retryWrites=true&w=majority")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client.get_database("dL")

collections = {
    "bookq": "Bookq",
    "mean": "mean_json"
}

def timeout_handling(max_time):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            if time.time() - start_time > max_time:
                raise TimeoutError(f"Function exceeded max time of {max_time} seconds")
            return result
        return wrapper
    return decorator

def load_data(collection_key, limit=MAX_ROWS_FOR_EDA):
    try:
        coll_name = collections.get(collection_key)
        if not coll_name:
            print(f"Collection {collection_key} not found")
            return None
        data = list(db[coll_name].find({}, {'_id': 0}).limit(limit))
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {collection_key}: {str(e)}")
        return None

@timeout_handling(MAX_EDA_TIME)
def perform_eda(df, name):
    try:
        if df is None or df.empty or len(df.columns) < 2:
            print(f"Not enough data for {name}")
            return False

        # Limit data size for EDA
        if len(df) > MAX_ROWS_FOR_EDA:
            df = df.sample(MAX_ROWS_FOR_EDA)

        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            print(f"Not enough numeric columns in {name}")
            return False

        plt.switch_backend('Agg')
        
        # Simplified Correlation Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".1f")
        plt.title(f'Correlation - {name}', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'static/corr_{name}.png', dpi=80, bbox_inches='tight')
        plt.close()

        # Simplified Pairplot (using smaller sample)
        sample_df = numeric_df.sample(min(100, len(numeric_df)))
        plt.figure(figsize=(10, 8))
        sns.pairplot(sample_df)
        plt.savefig(f'static/pairplot_{name}.png', dpi=80)
        plt.close()

        return True
    except Exception as e:
        print(f"EDA failed for {name}: {str(e)}")
        return False

def train_models():
    results = {}
    try:
        df_bookq = load_data("bookq", limit=500)
        df_mean = load_data("mean", limit=500)

        for collection_name, df in [("bookq", df_bookq), ("mean", df_mean)]:
            if df is None or df.empty:
                results[collection_name] = {"error": "No data available"}
                continue

            try:
                df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
                
                if 'Temperature' not in df.columns or 'Gas Emission Value' not in df.columns:
                    results[collection_name] = {
                        "error": f"Missing required columns. Available: {df.columns.tolist()}"
                    }
                    continue

                X = df[["Temperature"]]
                y = df["Gas Emission Value"]

                models = {
                    "LinearRegression": LinearRegression(),
                    "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42),
                    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42)
                }

                metrics = {}
                for name, model in models.items():
                    try:
                        model.fit(X, y)
                        preds = model.predict(X)
                        metrics[name] = {
                            "MAE": round(mean_absolute_error(y, preds), 2),
                            "R2": round(r2_score(y, preds), 4),
                            "RMSE": round(np.sqrt(mean_squared_error(y, preds)), 2)
                        }
                    except Exception as e:
                        metrics[name] = {"error": str(e)}

                results[collection_name] = metrics
            except Exception as e:
                results[collection_name] = {"error": str(e)}

    except Exception as e:
        results = {"error": f"Training failed: {str(e)}"}

    return results

@app.route('/get_eda_results')
def get_eda_results():
    try:
        # Load data with limits
        df_bookq = load_data("bookq")
        df_mean = load_data("mean")

        if df_bookq is None or df_mean is None:
            return jsonify({"error": "Data loading failed"}), 500

        # Perform EDA with timeout protection
        eda_results = {
            "bookq": perform_eda(df_bookq, "bookq"),
            "mean": perform_eda(df_mean, "mean")
        }

        # Train models
        model_metrics = train_models()

        response = {
            "status": "success",
            "eda_results": {
                "correlation_images": {
                    "bookq": "/static/corr_bookq.png",
                    "mean": "/static/corr_mean.png"
                },
                "pairplots": {
                    "bookq": "/static/pairplot_bookq.png",
                    "mean": "/static/pairplot_mean.png"
                }
            },
            "model_metrics": model_metrics
        }

        return jsonify(response)

    except TimeoutError as e:
        return jsonify({"error": "EDA processing timed out", "details": str(e)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ... (keep your existing route handlers for static files, correlation, pairplot)

@app.route('/')
def home():
    return """
    <h2>Flask App is Running!</h2>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/get_eda_results">/get_eda_results</a> - Get EDA results and model metrics</li>
        <li>/correlation/[bookq|mean] - Get correlation image</li>
        <li>/pairplot/[bookq|mean] - Get pairplot image</li>
    </ul>
    """

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/correlation/<name>')
def get_correlation_image(name):
    path = f'corr_{name}.png'
    try:
        return send_from_directory('static', path, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Image not found. Please run /get_eda_results first"}), 404

@app.route('/pairplot/<name>')
def get_pairplot_image(name):
    path = f'pairplot_{name}.png'
    try:
        return send_from_directory('static', path, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Image not found. Please run /get_eda_results first"}), 404

@app.route('/get_eda_results')
def get_eda_results():
    try:
        # Ensure static directory exists
        os.makedirs("static", exist_ok=True)
        
        # Load data
        df_bookq = load_data("bookq")
        df_mean = load_data("mean")

        if df_bookq is None or df_mean is None:
            return jsonify({"error": "Data loading failed"}), 500

        # Perform EDA
        perform_eda(df_bookq, "bookq")
        perform_eda(df_mean, "mean")

        # Train models and get metrics
        model_metrics = train_models()

        return jsonify({
            "status": "success",
            "eda_results": {
                "correlation_image_bookq": "/static/corr_bookq.png",
                "correlation_image_mean": "/static/corr_mean.png",
                "pairplot_bookq": "/static/pairplot_bookq.png",
                "pairplot_mean": "/static/pairplot_mean.png"
            },
            "model_metrics": model_metrics
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)