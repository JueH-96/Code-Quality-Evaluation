# YOUR CODE HERE
N = int(input())

# Find the water station positions before and after N
lower_station = (N // 5) * 5
upper_station = lower_station + 5

# Calculate distances to both stations
dist_to_lower = abs(N - lower_station)
dist_to_upper = abs(N - upper_station)

# Find the nearest station
if dist_to_lower <= dist_to_upper:
    nearest_station = lower_station
else:
    nearest_station = upper_station

print(nearest_station)