# YOUR CODE HERE
import bisect

# Read input
N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort B array for binary search
B.sort()

max_sum = -1

# For each gift for Aoki
for a in A:
    # Find the range of valid B values: [a-D, a+D]
    # Find the rightmost B that is <= a+D
    right_idx = bisect.bisect_right(B, a + D) - 1
    
    # Find the leftmost B that is >= a-D
    left_idx = bisect.bisect_left(B, a - D)
    
    # Check if there's any valid B in the range
    if left_idx <= right_idx and right_idx >= 0 and left_idx < M:
        # The maximum valid B is at right_idx
        max_sum = max(max_sum, a + B[right_idx])

print(max_sum)