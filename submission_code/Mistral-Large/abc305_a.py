import sys

def find_nearest_water_station(N):
    # Calculate the distance to the next water station
    next_station = ((N + 4) // 5) * 5
    # Calculate the distance to the previous water station
    prev_station = ((N - 1) // 5) * 5

    # Determine the nearest water station
    if N - prev_station <= next_station - N:
        return prev_station
    else:
        return next_station

# Read input from stdin
N = int(sys.stdin.read().strip())

# Find and print the nearest water station
print(find_nearest_water_station(N))