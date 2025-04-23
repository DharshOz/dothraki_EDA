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
from pandas.plotting import parallel_coordinates

# Flask App Config
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
os.makedirs("static", exist_ok=True)

# MongoDB Connection
MONGO_URI = "mongodb+srv://vgugan16:gugan2004@cluster0.qyh1fuo.mongodb.net/dL?retryWrites=true&w=majority"
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
            # Simple merge - assumes same number of records in same order
            df_mean['Temperature'] = df_bookq['Temperature'].values
            print("Successfully merged Temperature from bookq to mean")
        return df_mean
    except Exception as e:
        print(f"Error merging Temperature: {str(e)}")
        return df_mean


def perform_eda(df, name):
    try:
        if df.empty or len(df.columns) < 2:
            print(f"Not enough data for {name}")
            return False

        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            print(f"Not enough numeric columns in {name}")
            return False

        # Correlation Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f'Correlation - {name}')
        plt.tight_layout()
        plt.savefig(f'static/corr_{name}.png')
        plt.close()

        # Pairplot (sample first 100 rows if large dataset)
        sample_size = min(100, len(numeric_df))
        plt.figure(figsize=(12, 10))
        sns.pairplot(numeric_df.sample(sample_size))
        plt.suptitle(f'Pairplot - {name} (Sample of {sample_size})', y=1.02)
        plt.savefig(f'static/pairplot_{name}.png')
        plt.close()

        # Parallel Coordinates Plot (piplot)
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(15, 8))
            parallel_coordinates(numeric_df.sample(min(100, len(numeric_df))),
                                 'Gas Emission Value' if 'Gas Emission Value' in numeric_df.columns else
                                 numeric_df.columns[0],
                                 color=('#556270', '#4ECDC4', '#C7F464', '#FF6B6B', '#C44D58'))
            plt.title(f'Parallel Coordinates Plot - {name}')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'static/piplot_{name}.png')
            plt.close()

        # Distribution Plots
        for col in numeric_df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(numeric_df[col], kde=True)
            plt.title(f'Distribution of {col} - {name}')
            plt.tight_layout()
            plt.savefig(f'static/dist_{name}_{col}.png')
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

    # Handle temperature merging more carefully
    if df_mean is not None and not df_mean.empty and df_bookq is not None and not df_bookq.empty:
        if 'Temperature' not in df_mean.columns and 'Temperature' in df_bookq.columns:
            # Find the smaller dataset size
            min_length = min(len(df_mean), len(df_bookq))

            # Truncate both datasets to the same size
            df_mean = df_mean.head(min_length)
            df_bookq = df_bookq.head(min_length)

            # Now safely copy the temperature values
            df_mean['Temperature'] = df_bookq['Temperature'].values
            print(f"Successfully merged Temperature (using {min_length} records)")

    for collection_name in ["bookq", "mean"]:
        try:
            # Use the appropriate dataframe
            df = df_bookq if collection_name == "bookq" else df_mean

            if df is None or df.empty:
                results[collection_name] = {"error": "No data available"}
                continue

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


# Routes
@app.route('/')
def home():
    return "<h2>Flask App is Running!</h2>"


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)


@app.route('/correlation/<name>')
def get_correlation_image(name):
    path = f'corr_{name}.png'
    try:
        return send_from_directory('static', path, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404


@app.route('/piplot/<name>')
def get_piplot_image(name):
    path = f'piplot_{name}.png'
    try:
        return send_from_directory('static', path, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404


@app.route('/pairplot/<name>')
def get_pairplot_image(name):
    path = f'pairplot_{name}.png'
    try:
        return send_from_directory('static', path, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404


@app.route('/distribution/<name>/<column>')
def get_distribution_image(name, column):
    path = f'dist_{name}_{column}.png'
    try:
        return send_from_directory('static', path, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404


@app.route('/get_eda_results')
def get_eda_results():
    try:
        df_bookq = load_data("bookq")
        df_mean = load_data("mean")

        if df_bookq is None or df_mean is None:
            return jsonify({"error": "Data loading failed"}), 500

        # Perform EDA before merging to see original correlations
        eda_bookq = perform_eda(df_bookq, "bookq")
        eda_mean = perform_eda(df_mean, "mean")

        model_metrics = train_models()

        return jsonify({
            "status": "success",
            "eda_results": {
                "correlation_images": {
                    "bookq": "/static/corr_bookq.png",
                    "mean": "/static/corr_mean.png"
                },
                "piplot_images": {
                    "bookq": "/static/piplot_bookq.png",
                    "mean": "/static/piplot_mean.png"
                },
                "pairplot_images": {
                    "bookq": "/static/pairplot_bookq.png",
                    "mean": "/static/pairplot_mean.png"
                },
                "distribution_images": {
                    "bookq": [f"/static/dist_bookq_{col}.png" for col in
                              df_bookq.select_dtypes(include=['number']).columns],
                    "mean": [f"/static/dist_mean_{col}.png" for col in
                             df_mean.select_dtypes(include=['number']).columns]
                }
            },
            "model_metrics": model_metrics
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/eda')
def serve_eda():
    return send_file('D:\mobile_app_development\cat2_project\smartCity\www\EDA.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)