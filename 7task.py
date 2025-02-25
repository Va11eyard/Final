import numpy as np
import pandas as pd

# Example dataset
data = pd.DataFrame({
    'Group': ['Mexican', 'Other'],
    'n': [121, 81],
    'x': [1500, 1500]
})

design_effect = {"Mexican": 3.09, "Other": 4.08}  # Example values, adjust as needed

results = {}

# Process each row in the dataframe
for _, row in data.iterrows():
    group = row['Group']
    n, x = row['n'], row['x']
    p_hat = x / n

    # Standard Error
    SE = np.sqrt((p_hat * (1 - p_hat)) / n)

    # Adjusted SE (Clustering)
    SE_adjusted = SE * np.sqrt(design_effect.get(group, 1))

    # Confidence Intervals
    CI = (p_hat - 1.96 * SE, p_hat + 1.96 * SE)
    CI_adjusted = (p_hat - 1.96 * SE_adjusted, p_hat + 1.96 * SE_adjusted)

    results[group] = {
        "Estimated Proportion": round(p_hat, 4),
        "Standard Error": round(SE, 4),
        "Adjusted SE": round(SE_adjusted, 4),
        "95% CI": (round(CI[0], 2), round(CI[1], 2)),
        "95% Adjusted CI": (round(CI_adjusted[0], 2), round(CI_adjusted[1], 2)),
        "Design Effect": design_effect.get(group, 1)
    }

# Display results
for group, values in results.items():
    print(f"{group}: {values}")
