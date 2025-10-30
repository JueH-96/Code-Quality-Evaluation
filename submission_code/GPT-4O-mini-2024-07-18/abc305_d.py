def sleep_log(N, A, Q, queries):
    # Create a list to store the sleep intervals
    sleep_intervals = []
    
    for i in range(1, N // 2 + 1):
        sleep_start = A[2 * i - 1]
        sleep_end = A[2 * i]
        sleep_intervals.append((sleep_start, sleep_end))
    
    results = []
    
    for l, r in queries:
        total_sleep = 0
        
        for start, end in sleep_intervals:
            if end <= l:
                continue
            if start >= r:
                break
            
            # Calculate the effective sleep time within the range [l, r]
            effective_start = max(start, l)
            effective_end = min(end, r)
            total_sleep += effective_end - effective_start
        
        results.append(total_sleep)
    
    return results

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N + 1]))
Q = int(data[N + 1])
queries = []

for i in range(Q):
    l = int(data[N + 2 + 2 * i])
    r = int(data[N + 3 + 2 * i])
    queries.append((l, r))

# Get results
results = sleep_log(N, A, Q, queries)

# Print results
for result in results:
    print(result)