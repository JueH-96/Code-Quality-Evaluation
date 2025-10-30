N = int(input())

# Calculate the nearest water station
nearest_water_station = (N // 5) * 5
next_water_station = nearest_water_station + 5

# Determine which water station is closer
if N - nearest_water_station <= next_water_station - N:
    print(nearest_water_station)
else:
    print(next_water_station)