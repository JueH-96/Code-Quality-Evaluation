# YOUR CODE HERE
N = int(input())

# Find the water station before or at position N
station_before = (N // 5) * 5

# Find the water station after position N
station_after = station_before + 5

# If N is already at a water station
if N % 5 == 0:
    print(N)
else:
    # Calculate distances to both stations
    dist_before = N - station_before
    dist_after = station_after - N
    
    # Choose the closer one
    if dist_before <= dist_after:
        print(station_before)
    else:
        print(station_after)