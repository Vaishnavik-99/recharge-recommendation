import pickle
from preprocess import load_and_preprocess
from sklearn.preprocessing import StandardScaler

def train_model(data_path, model_path):
    df = load_and_preprocess(data_path)
    
    # Features for similarity
    features = ['Price','Validity_Days','Talktime','Data_GB','SMS']
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    # Save scaled data and original df for recommendation
    with open(model_path, 'wb') as f:
        pickle.dump({'df': df, 'scaled': df_scaled, 'features': features}, f)
    
    print("Model/data prepared and saved.")

if __name__ == "__main__":
    train_model("../data/recharge_plans.csv", "../models/recharge_model.pkl")
