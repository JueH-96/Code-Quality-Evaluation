import bisect

N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Precompute cumulative sleep time at each time point in A
cum_sleep = [0] * N
for i in range(1, N):
    cum_sleep[i] = cum_sleep[i-1]
    if i % 2 == 0:  # A[i] is a wake-up time (end of sleep session)
        cum_sleep[i] += A[i] - A[i-1]

# Function to compute total sleep time from 0 to t
def sleep_until(t):
    # Find the position of t in A using binary search
    idx = bisect.bisect_right(A, t) - 1
    
    if idx < 0:
        return 0
    
    result = cum_sleep[idx]
    
    # If idx is odd, we're in the middle of a sleep session
    if idx % 2 == 1:
        # We're sleeping from A[idx] to A[idx+1]
        # Add the partial sleep from A[idx] to t
        if idx + 1 < N:
            result += min(t, A[idx + 1]) - A[idx]
    
    return result

for _ in range(Q):
    l, r = map(int, input().split())
    print(sleep_until(r) - sleep_until(l))