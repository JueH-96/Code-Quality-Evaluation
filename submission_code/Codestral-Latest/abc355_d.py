import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
intervals = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

# Sort intervals by their starting points
intervals.sort()

count = 0
active_intervals = []

for i in range(N):
    # Remove intervals from active_intervals that do not intersect with the current interval
    while active_intervals and active_intervals[0][1] <= intervals[i][0]:
        active_intervals.pop(0)

    # Add the current interval to active_intervals
    active_intervals.append(intervals[i])

    # Count the number of intersections
    count += len(active_intervals) - 1

print(count)