import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import sqlite3
import pandas as pd

def check_for_nans(df):
    """Check if any NaNs are present in the DataFrame."""
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values found!")
        print(df.isnull().sum())  # Print the number of missing values per column
    else:
        print("No missing values found in the dataset.")

# Assuming preprocess_data is your preprocessing function
def preprocess_data(df):
    """Preprocess data: handle missing values, encode categories, and scale numerical features."""
    
    # Step 1: Handle missing values
    df['room'] = df['room'].fillna('Unknown')
    df['price'] = df['price'].replace({r'SGD\$': '', r'USD\$': ''}, regex=True).astype(float)
    df['price'] = df.groupby('room')['price'].transform(lambda x: x.fillna(x.median()))
    df.dropna(subset=['no_show', 'branch', 'booking_month', 'arrival_month', 'arrival_day',
                    'checkout_month', 'checkout_day', 'country', 'first_time', 'platform',
                    'num_adults', 'num_children'], inplace=True)    
    
    # Step 2: Convert data types
    df['num_adults'] = df['num_adults'].replace({'one': 1, 'two': 2})
    df['num_adults'] = pd.to_numeric(df['num_adults'], errors='coerce')
    df['num_adults'].fillna(df['num_adults'].median(), inplace=True)
    df['num_children'] = pd.to_numeric(df['num_children'], errors='coerce')

    df['arrival_month'] = df['arrival_month'].str.strip().str.capitalize()

    df['total_guests'] = df['num_adults'] + df['num_children']

    df['checkout_weekend'] = df['checkout_day'].apply(lambda x: 1 if x in [5, 6] else 0)

    # Step 3: Create 'price_category' based on quantiles
    df['price_category'] = pd.qcut(df['price'], q=4, labels=['low', 'medium', 'high', 'very_high'])
    
    # Step 4: Drop unnecessary columns
    df.drop(columns=['booking_id'], inplace=True)

    # Step 5: Create interaction feature
    df['price_guests_interaction'] = df['price_category'].astype(str) + "_" + df['total_guests'].astype(str)
    
    # Step 6: Encode categorical features (LabelEncoding for some, OneHotEncoding for 'price_category')
    label_encoder = LabelEncoder()
    categorical_cols = ['branch', 'booking_month', 'arrival_month', 'checkout_month', 'country',
       'first_time', 'room', 'platform', 'price_guests_interaction']
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # OneHotEncode 'price_category'
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    price_category_encoded = one_hot_encoder.fit_transform(df[['price_category']])
    price_category_df = pd.DataFrame(price_category_encoded, columns=one_hot_encoder.get_feature_names_out(['price_category']))
    
    # Concatenate the encoded 'price_category' columns back into the dataframe
    df = pd.concat([df, price_category_df], axis=1)
    df.drop(columns=['price_category'], inplace=True)
    df = df.dropna(axis=0)  # Drops rows with any NaN values

    # Step 7: Scale numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('no_show', errors='ignore')
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Step 8: Split into training and testing sets
    target = 'no_show'  # Define the target variable
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, df  # Return the final df as well for checking

# Assuming 'df' is your DataFrame (replace this with your actual DataFrame)
def fetch_data(db_file, table_name):
    """Connect to SQLite database and fetch data."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df
df = fetch_data("data/noshow.db", "noshow")
# Run preprocessing and get the processed dataframe
X_train, X_test, y_train, y_test, df_processed = preprocess_data(df)

# Check for NaNs after preprocessing
check_for_nans(df_processed)
