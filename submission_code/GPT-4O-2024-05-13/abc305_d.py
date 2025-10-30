# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = []
for i in range(Q):
    l = int(data[N+2 + 2*i])
    r = int(data[N+3 + 2*i])
    queries.append((l, r))

# Precompute sleep intervals
sleep_intervals = []
for i in range(1, N, 2):
    sleep_intervals.append((A[i], A[i+1]))

# Function to calculate sleep time in a given interval
def sleep_time_in_interval(l, r):
    total_sleep = 0
    for start, end in sleep_intervals:
        if end <= l or start >= r:
            continue
        sleep_start = max(start, l)
        sleep_end = min(end, r)
        total_sleep += max(0, sleep_end - sleep_start)
    return total_sleep

# Answer each query
results = []
for l, r in queries:
    results.append(sleep_time_in_interval(l, r))

# Print results
for result in results:
    print(result)