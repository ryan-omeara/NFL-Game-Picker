import numpy as np
import pandas as pd

#Used to Generate the Weekly Predictions and Weighted Probabilities for scoring
def generate_inference_dataframe(week, year, schedules, scaler, features) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = schedules[schedules['year'] == year]
    if (week != None):
        df = df[df['week'] == week]
    df = df.reset_index(drop=True)
    df_scaled = scaler.transform(df[features])
    return df, df_scaled

def generate_predictions(model, df_scaled) -> np.ndarray:
    weekly_probs = model.predict_proba(df_scaled)[:, 1]
    return weekly_probs

def print_predictions(df, probabilites):
    for index, row in df.iterrows():
        print(f"Game: {row['away_team']} @ {row['home_team']}, Predicted Win Probability for Home Team: {probabilites[index]:.4f}")

def generate_picks(df, probabilities):
    #Account for points based on spread. 1pt for favorite, 2 point for underdog. 3 points for 7 pt underdog
    #add cols for homeIsFavorite (bool), 7ptUnderdog (bool), evHome (float), evAway (float), bestEV ("Home or Away")
    results = pd.DataFrame()
    #Favorites
    results.insert(0, 'favorite', np.where(
        df['spread_line'] >= 0, 
        df['home_team'], 
        df['away_team'])   )
    #Underdogs
    results['underdog'] = np.where(
        df['spread_line'] >= 0, 
        df['away_team'], 
        df['home_team'])
    #Spread
    results['spread'] = np.where(
        df['home_team'] == results['favorite'],
        df['spread_line'],
        df['spread_line'] * -1,
    )
    #Favorite Win Prob
    results['favorite_win_prob'] = np.where(
        df['home_team'] == results['favorite'],
        probabilities,
        1 - probabilities
    )
    #Underdog Win Prob
    results['underdog_win_prob'] = 1 - results['favorite_win_prob']
    #EV Home
    results['favorite_ev'] = results['favorite_win_prob'] * 1
    #Is Underdog 3pt (7+ point spread)
    results['is_3pt_underdog'] = results['spread'] >= 7
    #Ev Away
    results['underdog_ev'] = np.where(
        results['is_3pt_underdog'],
        results['underdog_win_prob'] * 3,
        results['underdog_win_prob'] * 2
    )
    #Team Pick (highest EV)
    results['pick'] = np.where(
        results['favorite_ev'] >= results['underdog_ev'],
        results['favorite'],
        results['underdog']
    )
    return results
