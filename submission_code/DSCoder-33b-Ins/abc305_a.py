# YOUR CODE HERE

N = int(input())

# Calculate the nearest water station
if N % 5 < 3:
    nearest_station = N - (N % 5)
else:
    nearest_station = N + (5 - (N % 5))

print(nearest_station)