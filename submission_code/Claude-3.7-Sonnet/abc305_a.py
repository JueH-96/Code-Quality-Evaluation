# YOUR CODE HERE
N = int(input())

# Calculate the positions of the two potential nearest water stations
lower_station = (N // 5) * 5
upper_station = min(100, ((N + 4) // 5) * 5)

# Determine which station is closer to Takahashi
distance_to_lower = N - lower_station
distance_to_upper = upper_station - N

if distance_to_lower <= distance_to_upper:
    nearest_station = lower_station
else:
    nearest_station = upper_station

print(nearest_station)