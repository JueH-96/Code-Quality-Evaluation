import sys

# Read input from standard input
# N: total number of stations (not directly used in the core logic, but defines station range)
# X: starting station
# Y: destination station
# Z: station to check if it's a stop during the travel
N, X, Y, Z = map(int, sys.stdin.readline().split())

# Determine the minimum and maximum station numbers between X and Y.
# This defines the full range of stations involved in the travel.
min_station_on_path = min(X, Y)
max_station_on_path = max(X, Y)

# The problem asks if the train "stops at" station Z "during" this travel.
# Based on the sample explanation ("After departing from station X, the train stops at ...")
# and the constraint that X, Y, Z are distinct:
# - The starting station X is not counted as a stop during the travel.
# - The destination station Y is typically counted as a stop, but since Z cannot be Y (due to distinct constraint),
#   we only care about intermediate stops.
# - Thus, Z must be a station strictly between X and Y on the path.
# This means Z must be greater than the minimum of X and Y, and less than the maximum of X and Y.

if min_station_on_path < Z < max_station_on_path:
    print("Yes")
else:
    print("No")