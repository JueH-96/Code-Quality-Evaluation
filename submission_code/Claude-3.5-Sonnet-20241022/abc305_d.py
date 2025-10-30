N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Create list of sleep intervals
sleep_intervals = []
for i in range(1, N-1, 2):
    sleep_intervals.append((A[i], A[i+1]))

def get_overlap(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start)

# Process each query
for _ in range(Q):
    l, r = map(int, input().split())
    query_interval = (l, r)
    
    total_sleep = 0
    for sleep_interval in sleep_intervals:
        overlap = get_overlap(query_interval, sleep_interval)
        total_sleep += overlap
        
    print(total_sleep)