import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#this is for current data
matches = pd.read_csv("assets/soccer-matches.csv")

matches["Date Time (US Eastern)"] = pd.to_datetime(matches["Date Time (US Eastern)"])
#advantages and disadvantages to consider when teams play against each other
matches["homeAway_code"] = matches["homeAway"].astype("category").cat.codes
matches["opponent_code"] = np.where(
    matches["Team"] == matches["Home Team"],
    matches["Away Team"].astype("category").cat.codes,
    matches["Home Team"].astype("category").cat.codes
)
""" 
info to train the model
2. On target shots
3. Goal (expected goals)
4. Possession 
5. Pass completsion 
"""
#determine the match result
matches["target"] = np.where(
    matches["Home Goal"] == matches["Away Goal"], 2,  # Draw condition
    np.where(
        (matches["Home Goal"] > matches["Away Goal"]) & (matches["Team"] == matches["Home Team"]), 1,  # Home win for home team
        np.where(
            (matches["Home Goal"] > matches["Away Goal"]) & (matches["Team"] != matches["Home Team"]), 0,  # Home win for away team
            np.where(
                (matches["Home Goal"] < matches["Away Goal"]) & (matches["Team"] == matches["Away Team"]), 1,  # Away win for away team
                0  # Away win for home team
            )
        )
    )
)
""" matches["Target"] = (matches["result"]=="W").astype("int") """

#began using machine learning 
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
predictors = ["homeAway_code", "opponent_code", "hour", "day_code"]
train = matches[matches["Date Time (US Eastern)"] < '2024-10-6']
test = matches[matches["Date Time (US Eastern)"] > '2024-10-6']
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])
accuracy = accuracy_score(test["target"], preds)
accuracy

