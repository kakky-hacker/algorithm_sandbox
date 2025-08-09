from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt 
  
seoul_bike_sharing_demand = fetch_ucirepo(id=560) 
df = seoul_bike_sharing_demand.data.features 
df['Temperature'] = df['Hour'].astype(int)
daily_bike_rentals = df.groupby('Temperature')['Rented Bike Count'].sum()

plt.figure(figsize=(10, 6))
daily_bike_rentals.plot()
plt.title('Daily Rented Bike Count')
plt.xlabel('Temperature')
plt.ylabel('Rented Bike Count')
plt.grid(True)
plt.show()