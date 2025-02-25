import pandas as pd
import numpy as np

df = pd.read_excel("Question1_Final_CP.xlsx")


# Calculate the total population size
N = len(df)

# Group by Stratum
strata_group = df.groupby('Stratum')

# Compute the mean and size for each stratum
stratum_means = strata_group['Corruption level Rating Score (0-100)'].mean()
stratum_sizes = strata_group.size()

# Compute the weight for each stratum (Wh)
stratum_weights = stratum_sizes / N

# Output results
print(f"Stratum Weights (W_h):\n{stratum_weights}")


#4.2
print("Compute a mean:", df['Corruption level Rating Score (0-100)'].mean().round(2))


df['Y'] = (df['Corruption level Rating Score (0-100)']-52.28)**2

stratums = df['Stratum'].unique()
sh_values = {}

for stratum in stratums:
    sh_values[stratum] = df[df['Stratum'] == stratum]['Y'].sum() / float(len(df[df['Stratum'] == stratum]['Y']) - 1)

#4.3
import math
standart_strat = math.sqrt(((0.25**2*sh_values['North America']/8)+
                            (0.25**2*sh_values['South America']/8)+
                            (0.25**2*sh_values['Europe']/8)+
                            (0.25**2*sh_values['Central Asia']/8)))

print("Compute a standard error for Stratified part:", round(standart_strat, 2))

#4.4
standart_error_srs = df['Corruption level Rating Score (0-100)'].sem()
d = round(standart_strat / standart_error_srs, 2)
print("Compute d-value:", d)

#4.5
d_squared = round(d*d, 2)
print("Compute d-squared:", d_squared)

#4.6
neff = round(32/(d_squared), 2)
print("Compute Neff:", neff)