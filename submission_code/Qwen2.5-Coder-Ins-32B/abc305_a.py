# YOUR CODE HERE
import sys

def find_nearest_water_station(N):
    # Water stations are at every 5 km, so we can find the nearest one by rounding N to the nearest multiple of 5
    lower_station = (N // 5) * 5
    upper_station = lower_station + 5
    
    # Calculate the distance to the lower and upper water stations
    distance_to_lower = abs(N - lower_station)
    distance_to_upper = abs(N - upper_station)
    
    # Return the position of the nearest water station
    if distance_to_lower <= distance_to_upper:
        return lower_station
    else:
        return upper_station

# Read input from stdin
N = int(sys.stdin.read().strip())

# Find and print the nearest water station
print(find_nearest_water_station(N))