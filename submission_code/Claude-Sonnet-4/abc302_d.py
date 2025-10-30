import bisect

N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort both arrays
A.sort()
B.sort()

max_sum = -1

# For each gift for Aoki
for a in A:
    # Find the range of valid gifts for Snuke
    # We need |a - b| <= D, which means a - D <= b <= a + D
    
    # Find the leftmost position where b >= a - D
    left_idx = bisect.bisect_left(B, a - D)
    
    # Find the rightmost position where b <= a + D
    right_idx = bisect.bisect_right(B, a + D) - 1
    
    # If there's a valid range
    if left_idx <= right_idx and left_idx < M and right_idx >= 0:
        # Take the maximum value in this range (which is at right_idx since B is sorted)
        max_b = B[right_idx]
        current_sum = a + max_b
        max_sum = max(max_sum, current_sum)

print(max_sum)