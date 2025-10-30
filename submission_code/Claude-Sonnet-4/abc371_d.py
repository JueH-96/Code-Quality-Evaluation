import bisect

# Read input
N = int(input())
X = list(map(int, input().split()))
P = list(map(int, input().split()))
Q = int(input())

# Process queries
for _ in range(Q):
    L, R = map(int, input().split())
    
    # Find leftmost village with coordinate >= L
    left_idx = bisect.bisect_left(X, L)
    
    # Find rightmost village with coordinate <= R
    right_idx = bisect.bisect_right(X, R) - 1
    
    # Sum populations in range
    total = 0
    if left_idx <= right_idx and left_idx < N:
        for i in range(left_idx, min(right_idx + 1, N)):
            total += P[i]
    
    print(total)