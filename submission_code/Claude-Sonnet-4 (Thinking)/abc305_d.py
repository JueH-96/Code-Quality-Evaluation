import bisect

N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Extract sleep sessions
sleep_sessions = []
for i in range(1, (N - 1) // 2 + 1):
    start = A[2 * i - 1]
    end = A[2 * i]
    sleep_sessions.append((start, end))

# Precompute prefix sums
prefix_sleep = [0]
for start, end in sleep_sessions:
    prefix_sleep.append(prefix_sleep[-1] + (end - start))

# Extract end times for binary search
end_times = [end for start, end in sleep_sessions]

# Function to calculate total sleep time from 0 to t
def total_sleep_up_to(t):
    # Find the number of sessions that end at or before t
    idx = bisect.bisect_right(end_times, t)
    
    total = prefix_sleep[idx]
    
    # Check if t falls within the next session
    if idx < len(sleep_sessions):
        start, end = sleep_sessions[idx]
        if start < t:
            total += t - start
    
    return total

# Process each query
for _ in range(Q):
    l, r = map(int, input().split())
    answer = total_sleep_up_to(r) - total_sleep_up_to(l)
    print(answer)