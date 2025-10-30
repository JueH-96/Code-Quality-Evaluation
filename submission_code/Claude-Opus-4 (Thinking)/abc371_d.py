import bisect

# Read N
N = int(input())

# Read X coordinates
X = list(map(int, input().split()))

# Read P populations
P = list(map(int, input().split()))

# Create prefix sum array for efficient range sum
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + P[i]

# Read Q
Q = int(input())

# Process queries
for _ in range(Q):
    L, R = map(int, input().split())
    
    # Find the leftmost village with X >= L
    left_idx = bisect.bisect_left(X, L)
    
    # Find the rightmost village with X <= R
    right_idx = bisect.bisect_right(X, R) - 1
    
    # Calculate sum using prefix sum
    if left_idx <= right_idx:
        total = prefix_sum[right_idx + 1] - prefix_sum[left_idx]
    else:
        total = 0
    
    print(total)