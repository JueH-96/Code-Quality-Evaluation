# YOUR CODE HERE
import sys
input = sys.stdin.read

N = int(input().strip())

# Calculate the nearest water station
nearest_station = round(N / 5) * 5

print(nearest_station)