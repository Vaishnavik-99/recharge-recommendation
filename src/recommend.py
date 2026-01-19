import pickle
import numpy as np

def recommend_plan(model_path, user_input, top_n=3):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    df = data['df']
    scaled = data['scaled']
    features = data['features']
    
    # Scale user input
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[features])
    user_scaled = scaler.transform([user_input])
    
    # Compute similarity (Euclidean distance)
    distances = np.linalg.norm(scaled - user_scaled, axis=1)
    
    # Top N recommendations
    idx = distances.argsort()[:top_n]
    recommendations = df.iloc[idx]
    return recommendations

if __name__ == "__main__":
    # Example: user wants Price=250, Validity=28, Talktime=400, Data=1.5, SMS=100
    user_input = [250, 28, 400, 1.5, 100]
    result = recommend_plan("../models/recharge_model.pkl", user_input)
    print(result)
