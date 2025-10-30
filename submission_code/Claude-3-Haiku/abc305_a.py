import sys

N = int(input())

# Calculate the position of the nearest water station
nearest_station = (N // 5) * 5
if N % 5 > 2:
    nearest_station += 5

print(nearest_station)