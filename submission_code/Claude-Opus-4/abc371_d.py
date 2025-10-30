# YOUR CODE HERE
import bisect

# Read number of villages
N = int(input())

# Read village coordinates
X = list(map(int, input().split()))

# Read village populations
P = list(map(int, input().split()))

# Read number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    L, R = map(int, input().split())
    
    # Find the leftmost village with coordinate >= L
    left_idx = bisect.bisect_left(X, L)
    
    # Find the rightmost village with coordinate <= R
    right_idx = bisect.bisect_right(X, R) - 1
    
    # Calculate total population in the range
    total = 0
    for i in range(left_idx, right_idx + 1):
        if i >= 0 and i < N:
            total += P[i]
    
    print(total)