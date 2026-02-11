import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======================
# File path
# ======================
file_path = "accuracy_GPT4omini_8shot_10groups.jsonl"

# ======================
# 1. Load GSM-Symbolic accuracy data
# ======================
accuracies = []

# Read the JSONL file line by line
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())
        # Convert accuracy to percentage
        accuracies.append(record["accuracy"] * 100)

accuracies = np.array(accuracies)

# ======================
# 2. Compute statistics
# ======================
# Mean accuracy
mean_accuracy = accuracies.mean()

# Standard deviation (sample std, more appropriate for reporting)
std_accuracy = accuracies.std(ddof=1)

# GSM8K baseline accuracy (percentage)
gsm8k_accuracy = 96.0

# ======================
# 3. Plot distribution
# ======================
sns.set_theme(style="ticks")

# Color similar to the paper figure
color = "#2E6FBA"

plt.figure(figsize=(8, 6))

# Histogram with KDE
sns.histplot(
    accuracies,
    kde=True,
    bins=10,
    color=color,
    edgecolor=color,
    linewidth=1.5,
    alpha=0.25,
)

# ======================
# 4. Reference lines
# ======================
# GSM8K baseline
plt.axvline(
    x=gsm8k_accuracy,
    color="gray",
    linestyle="--",
    linewidth=2,
    label=f"GSM8K {gsm8k_accuracy:.1f}",
)

# GSM-Symbolic mean with standard deviation
plt.axvline(
    x=mean_accuracy,
    color=color,
    linestyle="--",
    linewidth=2,
    label=f"GSM-Symbolic {mean_accuracy:.1f} (±{std_accuracy:.1f})",
)

# ======================
# 5. Title and axis labels
# ======================
plt.title("GPT-4o-mini", fontsize=16)
plt.xlabel("GSM Symbolic Accuracy (%) - (8s CoT)", fontsize=13)
plt.ylabel("Frequency", fontsize=13)

# ======================
# 6. Legend
# ======================
plt.legend(frameon=False, fontsize=12, loc="upper right")

# ======================
# 7. Save and display the figure
# ======================
output_file = "accuracy_distribution_GPT4omini_symbolic.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
