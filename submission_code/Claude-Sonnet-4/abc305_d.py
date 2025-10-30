# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Extract sleep intervals
sleep_intervals = []
for i in range(1, n, 2):  # i = 1, 3, 5, ... (even indices in 0-based)
    sleep_start = a[i]
    sleep_end = a[i + 1]
    sleep_intervals.append((sleep_start, sleep_end))

q = int(input())
for _ in range(q):
    l, r = map(int, input().split())
    
    total_sleep = 0
    for sleep_start, sleep_end in sleep_intervals:
        # Calculate overlap between [sleep_start, sleep_end] and [l, r]
        overlap_start = max(sleep_start, l)
        overlap_end = min(sleep_end, r)
        
        if overlap_start < overlap_end:
            total_sleep += overlap_end - overlap_start
    
    print(total_sleep)