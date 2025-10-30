import bisect

N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort B for binary search
B.sort()

max_sum = -1

for a in A:
    # Find the range of valid B values
    # B[j] should be in [a - D, a + D]
    
    # Find the leftmost index where B[j] >= a - D
    left_idx = bisect.bisect_left(B, a - D)
    
    # Find the rightmost index where B[j] <= a + D
    right_idx = bisect.bisect_right(B, a + D) - 1
    
    # If there's a valid range
    if left_idx <= right_idx:
        # The maximum B[j] in the valid range is B[right_idx]
        max_sum = max(max_sum, a + B[right_idx])

print(max_sum)