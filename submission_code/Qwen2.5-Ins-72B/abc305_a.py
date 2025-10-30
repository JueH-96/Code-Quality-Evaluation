# YOUR CODE HERE
N = int(input())

# Calculate the nearest water station
nearest_station = (N + 2) // 5 * 5

# Check if the previous station is closer
if abs(N - nearest_station) > abs(N - (nearest_station - 5)):
    nearest_station -= 5

print(nearest_station)