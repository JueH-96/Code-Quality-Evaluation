import sys
import bisect

# Read input
input = sys.stdin.read
data = input().split()

# Parse the input
N = int(data[0])
A = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = [tuple(map(int, data[N+2+i*2:N+4+i*2])) for i in range(Q)]

# Precompute the sleep intervals
sleep_intervals = []
for i in range(1, N, 2):
    sleep_intervals.append((A[i], A[i+1]))

# Function to calculate the total sleep time in a given interval
def total_sleep_time(l, r):
    total_sleep = 0
    for start, end in sleep_intervals:
        if end <= l or start >= r:
            continue
        total_sleep += max(0, min(end, r) - max(start, l))
    return total_sleep

# Process each query
results = []
for l, r in queries:
    results.append(total_sleep_time(l, r))

# Output the results
for result in results:
    print(result)