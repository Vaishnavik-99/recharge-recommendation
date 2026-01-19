import pandas as pd

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    
    # Optional: Fill missing values
    df = df.fillna(0)
    
    # Optional: Convert categorical columns
    df['Category'] = df['Category'].astype('category')
    
    return df

if __name__ == "__main__":
    df = load_and_preprocess("../data/recharge_plans.csv")
    print(df.head())
