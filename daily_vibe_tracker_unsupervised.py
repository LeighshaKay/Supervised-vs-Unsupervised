import pandas as pd
import os
import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 🟢 Ask the user to rate their vibe
vibe = int(input("Rate your vibe today (1–10): "))

# 🗓 Get today's date
today = datetime.date.today().isoformat()

# 📁 Check if vibe_log.csv exists, else create new DataFrame
if os.path.exists("vibe_log.csv"):
    df = pd.read_csv("vibe_log.csv")
else:
    df = pd.DataFrame(columns=["date", "vibe"])

# ➕ Add today's entry
new_row = pd.DataFrame([{"date": today, "vibe": vibe}])
df = pd.concat([df, new_row], ignore_index=True)

# 💾 Save updated DataFrame to CSV
df.to_csv("vibe_log.csv", index=False)

# 📊 Print current log
print("\n📅 Current Vibe Log:")
print(df)

# 🤖 Unsupervised Learning: KMeans Clustering (if enough data)
X = df[["vibe"]]

if len(X) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    # 📈 Plot the clustered vibes
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    plt.figure(figsize=(10, 5))
    colors = ['red', 'orange', 'green']
    for i in range(3):
        cluster = df[df["cluster"] == i]
        plt.scatter(cluster["date"], cluster["vibe"], label=f"Cluster {i}", color=colors[i])

    plt.title("Unsupervised Daily Vibe Clusters (My Remix)")
    plt.xlabel("Date")
    plt.ylabel("Vibe Rating (1–10)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("\n⚠️ Not enough entries for clustering. Log at least 3 vibes first.")



