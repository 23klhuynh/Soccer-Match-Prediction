import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

matches = pd.read_csv("../assets/2023-2024.csv")
matches["Date Time (US Eastern)"] = pd.to_datetime(matches["Date Time (US Eastern)"])

#---advantages and disadvantages to consider when teams play against each other---
#matches["homeAway_code"] = pd.Series(np.where(matches["Team"] == matches["Home Team"], "home", "away")).astype("category").cat.codes  2022-2023
matches["homeAway_code"] = matches["homeAway"].astype("category").cat.codes
matches["opponent_code"] = np.where(
    matches["Team"] == matches["Home Team"],
    matches["Away Team"].astype("category").cat.codes,
    matches["Home Team"].astype("category").cat.codes
)
matches["hour"] = matches["Date Time (US Eastern)"].dt.hour.astype("Int64")
matches["day_code"] = matches["Date Time (US Eastern)"].dt.dayofweek

#---match result---
matches["target"] = np.where(
    matches["Home Goal"] == matches["Away Goal"], 2,  
    np.where(
        (matches["Home Goal"] > matches["Away Goal"]) & (matches["Team"] == matches["Home Team"]), 1,  
        np.where(
            (matches["Home Goal"] > matches["Away Goal"]) & (matches["Team"] != matches["Home Team"]), 0,  
            np.where(
                (matches["Home Goal"] < matches["Away Goal"]) & (matches["Team"] == matches["Away Team"]), 1,  
                0  
            )
        )
    )
)

#---finding the averages---
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date Time (US Eastern)")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    for col, new_col in zip(cols, new_cols):
        group[new_col] = rolling_stats[col]
    return group.dropna(subset=new_cols)

#---Add rolling averages---
columns = ["totalShots", "totalPasses"]

new_columns = [f"{c}_rolling" for c in columns]

matches_rolling = matches.groupby("Team").apply(lambda x: rolling_averages(x, columns, new_columns))
matches_rolling = matches_rolling.reset_index(drop=True)

#---Model Training and Evaluation---
def make_predictions(data, predictors, target_col='target'):
    train = data[data["Date Time (US Eastern)"] <= '2024-01-01']
    test = data[data["Date Time (US Eastern)"] > '2024-01-01']
    
    model = RandomForestClassifier(n_estimators=55, min_samples_split=20, random_state=1)
    model.fit(train[predictors], train[target_col])
    
    preds = model.predict(test[predictors])
    combine = pd.DataFrame({
        'actual': test[target_col],
        'prediction': preds
    }, index=test.index)
    
    accuracy = accuracy_score(test[target_col], preds)
    precision = precision_score(test[target_col], preds, average='macro', zero_division=0)
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")
    
    return combine, accuracy, precision
#
predictors = [   
    "homeAway_code", "opponent_code", "day_code", 
    "shotsOnTarget", "accuratePasses", "accurateCrosses", 
    "accurateLongBalls", "blockedShots", "effectiveTackles", 
    "effectiveClearance","foulsCommitted", "saves"
    ] + new_columns

#---predictions---
combined, accuracy, precision = make_predictions(matches_rolling, predictors)

# Merge results
combined = combined.merge(
    matches_rolling[["Date Time (US Eastern)", "Home Team", "Away Team", "target"]],
    left_index=True, right_index=True
)

""" print(combined.head())
print(precision)
 """

def predict_next_match(home_team, away_team, date_time):

    if home_team not in matches["Team"].unique() or away_team not in matches["Team"].unique():
        print("error")
        return
    
    #hour = pd.to_datetime(date_time).hour
    day_code = pd.to_datetime(date_time).dayofweek

    homeAway_code = 0 if home_team == home_team else 1
    opponent_code = matches["Team"].astype("category").cat.categories.get_loc(away_team)

    rolling_values = [0] * len(new_columns)  

    match_data = pd.DataFrame([{
        "homeAway_code": homeAway_code,
        "opponent_code": opponent_code,
        #"hour": hour,
        "day_code": day_code,
        **{col: val for col, val in zip(new_columns, rolling_values)}
        
    }])
    match_data = match_data[predictors]
    
    prediction = rf.predict(match_data)
    result = "Win" if prediction[0] == 1 else "Loss" if prediction[0] == 0 else "Draw"

    print(f"Prediction for {home_team} vs {away_team}: {result}")





