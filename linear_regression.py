import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Set styling for premium look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'image.cmap': 'viridis'
})

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'Housing.csv')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
for folder in [IMAGES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(folder, exist_ok=True)

def load_data(filepath):
    """Load the dataset from a CSV file."""
    print("Loading dataset...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def generate_dataset_preview(df):
    """Generate a clean image preview of the dataset dataframe head."""
    print("Generating dataset preview image...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Take first 10 rows and format columns
    df_preview = df.head(10).copy()
    
    # Find price and area columns (case-insensitive)
    price_col = next((c for c in df_preview.columns if c.lower() == 'price'), None)
    area_col = next((c for c in df_preview.columns if c.lower() == 'area'), None)
    
    if price_col:
        df_preview[price_col] = df_preview[price_col].apply(lambda x: f"${x:,.0f}")
    if area_col:
        df_preview[area_col] = df_preview[area_col].apply(lambda x: f"{x:,}")
    
    # Create matplotlib table
    table = ax.table(
        cellText=df_preview.values,
        colLabels=df_preview.columns,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    
    # Header cells styling
    for i, col in enumerate(df_preview.columns):
        cell = table[0, i]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50') # Sleek dark blue
        cell.set_edgecolor('#34495E')
        
    # Row cells styling (alternating colors)
    for row_idx in range(1, len(df_preview) + 1):
        face_color = '#F8F9FA' if row_idx % 2 == 0 else '#FFFFFF'
        for col_idx in range(len(df_preview.columns)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(face_color)
            cell.set_edgecolor('#E2E8F0')
            
    plt.title("Housing Dataset - First 10 Rows Preview", pad=20, weight='bold', color='#1A252C')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'dataset_preview.png'), bbox_inches='tight')
    plt.close()

def preprocess_dataset(df):
    """Preprocess raw housing data to numeric features."""
    print("Preprocessing dataset...")
    df_processed = df.copy()
    
    # Map binary features ('yes'/'no' -> 1/0)
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
            
    # Dummy encode furnishingstatus (categories: furnished, semi-furnished, unfurnished)
    if 'furnishingstatus' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['furnishingstatus'], drop_first=True, dtype=int)
        
    # Rename columns to a clean Title Case format
    rename_dict = {
        'price': 'Price',
        'area': 'Area',
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms',
        'stories': 'Stories',
        'mainroad': 'Main_Road_Access',
        'guestroom': 'Has_Guestroom',
        'basement': 'Has_Basement',
        'hotwaterheating': 'Has_Hot_Water',
        'airconditioning': 'Has_Air_Conditioning',
        'parking': 'Parking_Spaces',
        'prefarea': 'Preferred_Area',
        'furnishingstatus_semi-furnished': 'Furnished_Semi',
        'furnishingstatus_unfurnished': 'Furnished_No'
    }
    df_processed = df_processed.rename(columns=rename_dict)
    return df_processed

def generate_correlation_heatmap(df_processed):
    """Generate and save a correlation heatmap of the preprocessed dataset."""
    print("Generating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    corr = df_processed.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        square=True, 
        linewidths=.8, 
        cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "weight": "bold"}
    )
    plt.title("Feature Correlation Matrix Heatmap", pad=20, weight='bold', color='#1A252C')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close()

def train_and_evaluate(df_processed):
    """Train linear regression and save predictions, metrics, and plots."""
    print("Preparing features and targets...")
    X = df_processed.drop(columns=['Price'])
    y = df_processed['Price']
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    model_filepath = os.path.join(MODELS_DIR, 'linear_regression_model.pkl')
    print(f"Saving model artifacts to {model_filepath}...")
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    with open(model_filepath, 'wb') as f:
        pickle.dump(model_artifacts, f)
        
    # Predictions
    print("Evaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Save test predictions
    predictions_df = pd.DataFrame({
        'Actual_Price': y_test,
        'Predicted_Price': y_pred_test.round(0).astype(int),
        'Residual': (y_test - y_pred_test).round(0).astype(int)
    }).reset_index(drop=True)
    
    predictions_filepath = os.path.join(OUTPUTS_DIR, 'predictions.csv')
    predictions_df.to_csv(predictions_filepath, index=False)
    print(f"Saved predictions to {predictions_filepath}")
    
    # Evaluation Metrics
    mae = metrics.mean_absolute_error(y_test, y_pred_test)
    mse = metrics.mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred_test)
    
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    
    metrics_content = (
        "Linear Regression Model Evaluation Metrics\n"
        "===========================================\n"
        f"Train R² Score:         {r2_train:.4f}\n"
        f"Train MAE:              ${mae_train:,.2f}\n"
        "-------------------------------------------\n"
        f"Test R² Score (R-sq):   {r2:.4f}\n"
        f"Test MAE:               ${mae:,.2f}\n"
        f"Test MSE:               {mse:,.2f}\n"
        f"Test RMSE:              ${rmse:,.2f}\n"
        "===========================================\n"
        "Feature Coefficients (Scaled Features):\n"
    )
    for name, coef in zip(feature_names, model.coef_):
        metrics_content += f"  {name:25s}: {coef:,.2f}\n"
    metrics_content += f"Intercept:                  {model.intercept_:,.2f}\n"
    
    metrics_filepath = os.path.join(OUTPUTS_DIR, 'evaluation_metrics.txt')
    with open(metrics_filepath, 'w') as f:
        f.write(metrics_content)
    print(f"Saved evaluation metrics to {metrics_filepath}")
    print(metrics_content)
    
    # --- Generate Plots ---
    
    # 1. Regression Line Plot (Price vs Area)
    print("Generating regression line plot...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_processed, x='Area', y='Price', alpha=0.6, color='#2980B9', edgecolor='w', label='Actual Data')
    sns.regplot(data=df_processed, x='Area', y='Price', scatter=False, color='#C0392B', label='Regression Fit Line')
    plt.title("House Price vs. Area (sq ft) with Regression Line", weight='bold', color='#1A252C', pad=15)
    plt.xlabel("Area (sq ft)")
    plt.ylabel("Price ($)")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'regression_line.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Actual vs Predicted Plot
    print("Generating actual vs predicted plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7, color='#27AE60', edgecolor='w', label='Predicted vs Actual')
    
    max_val = max(y_test.max(), y_pred_test.max())
    min_val = min(y_test.min(), y_pred_test.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='#C0392B', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')
    
    plt.title("Actual vs. Predicted House Prices", weight='bold', color='#1A252C', pad=15)
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'actual_vs_predicted.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Residual Plot
    print("Generating residual plot...")
    plt.figure(figsize=(8, 6))
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.7, color='#8E44AD', edgecolor='w')
    plt.axhline(y=0, color='#C0392B', linestyle='--', linewidth=2)
    
    plt.title("Residual Plot (Homoscedasticity Check)", weight='bold', color='#1A252C', pad=15)
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residuals ($)")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'residual_plot.png'), bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance Plot
    print("Generating feature importance plot...")
    plt.figure(figsize=(9, 6))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=True)
    
    colors = ['#E74C3C' if x < 0 else '#2ECC71' for x in importance_df['Coefficient']]
    bars = plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, edgecolor='none', height=0.6)
    
    for bar in bars:
        width = bar.get_width()
        x_pos = width * 1.03 if width > 0 else width - (abs(width) * 0.08)
        ha_align = 'left' if width > 0 else 'right'
        plt.text(
            x_pos, 
            bar.get_y() + bar.get_height()/2, 
            f"${width:,.0f}", 
            va='center', 
            ha=ha_align, 
            fontsize=8, 
            weight='bold', 
            color='#2C3E50'
        )
        
    plt.axvline(x=0, color='#7F8C8D', linestyle='-', linewidth=0.8)
    plt.title("Feature Impact on House Price (Scaled Coefficients)", weight='bold', color='#1A252C', pad=20)
    plt.xlabel("Coefficient Value (Change in Price per 1 Std Dev change in Feature)")
    plt.ylabel("Feature")
    
    max_coeff = max(abs(importance_df['Coefficient']))
    plt.xlim(-max_coeff * 1.25, max_coeff * 1.25)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), bbox_inches='tight')
    plt.close()
    
    print("All diagnostic plots successfully saved to the images directory.")

if __name__ == '__main__':
    try:
        df = load_data(DATASET_PATH)
        generate_dataset_preview(df)
        df_processed = preprocess_dataset(df)
        generate_correlation_heatmap(df_processed)
        train_and_evaluate(df_processed)
        print("\nPipeline execution complete!")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
