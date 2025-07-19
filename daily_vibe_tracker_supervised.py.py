import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import os

# Create/load CSV file
filename = "vibe_data.csv"
if os.path.exists(filename):
    df = pd.read_csv(filename)
else:
    df = pd.DataFrame(columns=["date", "vibe"])

# Ask for today’s vibe
today = datetime.date.today().isoformat()
vibe = int(input("Rate your vibe today (1–10): "))

# Append and save
new_entry = pd.DataFrame([{"date": today, "vibe": vibe}])
df = pd.concat([df, new_entry], ignore_index=True)
df.to_csv(filename, index=False)
print("✅ Vibe saved!")

# Predict future vibes if enough data
if len(df) >= 2:
    df["day"] = range(len(df))
    X = df[["day"]]
    y = df["vibe"]

    model = LinearRegression().fit(X, y)
    next_day = [[len(df)]]
    prediction = model.predict(next_day)[0]

    print(f"🔮 Predicted vibe for tomorrow: {prediction:.2f}")

    # Plot
    plt.plot(df["day"], df["vibe"], label="Actual")
    plt.plot(df["day"], model.predict(X), label="Prediction", linestyle='--')
    plt.xlabel("Days")
    plt.ylabel("Vibe")
    plt.legend()
    plt.title("Vibe Tracker - Supervised")
    plt.show()
else:
    print("📊 Need at least 2 days of data to predict.")

