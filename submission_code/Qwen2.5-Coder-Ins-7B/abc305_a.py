# YOUR CODE HERE
import sys

N = int(sys.stdin.read().strip())

# Calculate the nearest water station
nearest_station = round(N / 5) * 5

print(nearest_station)