def max_distinct_sum(N, A):
    # Count distinct integers in the left part
    left_count = [0] * N
    right_count = [0] * N
    
    # Using a set to track distinct integers in the left part
    seen_left = set()
    for i in range(N):
        seen_left.add(A[i])
        left_count[i] = len(seen_left)
    
    # Using a set to track distinct integers in the right part
    seen_right = set()
    for i in range(N-1, -1, -1):
        seen_right.add(A[i])
        right_count[i] = len(seen_right)
    
    # Now calculate the maximum sum of distinct counts
    max_sum = 0
    for i in range(N-1):
        max_sum = max(max_sum, left_count[i] + right_count[i+1])
    
    return max_sum

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

result = max_distinct_sum(N, A)
print(result)