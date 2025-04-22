import os
import matplotlib
matplotlib.use('Agg')
from flask import Flask, jsonify, send_file, send_from_directory
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

# Flask App Config
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
os.makedirs("static", exist_ok=True)

# MongoDB Connection - Using environment variable for production safety
MONGO_URI = os.environ.get('MONGO_URI', "mongodb+srv://vgugan16:gugan2004@cluster0.qyh1fuo.mongodb.net/dL?retryWrites=true&w=majority")
client = MongoClient(MONGO_URI)
db = client["dL"]

collections = {
    "bookq": "Bookq",
    "mean": "mean_json"
}

def load_data(collection_key):
    try:
        coll_name = collections.get(collection_key)
        if not coll_name:
            print(f"Collection {collection_key} not found")
            return None
        data = list(db[coll_name].find({}, {'_id': 0}))
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {collection_key}: {str(e)}")
        return None

def merge_temperature(df_mean, df_bookq):
    """Merge Temperature from bookq to mean dataset"""
    try:
        if 'Temperature' not in df_mean.columns and 'Temperature' in df_bookq.columns:
            # Find the smaller dataset size
            min_length = min(len(df_mean), len(df_bookq))
            
            # Truncate both datasets to the same size
            df_mean = df_mean.head(min_length)
            df_bookq = df_bookq.head(min_length)
            
            # Now safely copy the temperature values
            df_mean['Temperature'] = df_bookq['Temperature'].values
            print(f"Successfully merged Temperature (using {min_length} records)")
        return df_mean
    except Exception as e:
        print(f"Error merging Temperature: {str(e)}")
        return df_mean

def perform_eda(df, name):
    try:
        if df is None or df.empty or len(df.columns) < 2:
            print(f"Not enough data for {name}")
            return False

        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            print(f"Not enough numeric columns in {name}")
            return False

        # Ensure we're using the Agg backend
        plt.switch_backend('Agg')
        
        # Correlation Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f'Correlation - {name}')
        plt.tight_layout()
        plt.savefig(f'static/corr_{name}.png')
        plt.close()

        # Pairplot
        plt.figure(figsize=(12, 8))
        sns.pairplot(numeric_df)
        plt.title(f'Pairplot - {name}')
        plt.tight_layout()
        plt.savefig(f'static/pairplot_{name}.png')
        plt.close()

        return True
    except Exception as e:
        print(f"EDA failed for {name}: {str(e)}")
        return False

def train_models():
    results = {}

    # Load both datasets first
    df_bookq = load_data("bookq")
    df_mean = load_data("mean")

    # Handle temperature merging
    if df_mean is not None and df_bookq is not None:
        df_mean = merge_temperature(df_mean, df_bookq)

    for collection_name in ["bookq", "mean"]:
        try:
            # Use the appropriate dataframe
            df = df_bookq if collection_name == "bookq" else df_mean

            if df is None or df.empty:
                results[collection_name] = {"error": "No data available"}
                continue

            # Convert to numeric and drop non-numeric columns
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
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "RandomForest": RandomForestRegressor(random_state=42)
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

    return results

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