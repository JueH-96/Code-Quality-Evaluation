# YOUR CODE HERE
import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the coordinates
A.sort()

# Initialize variables
max_gifts = 0
left = 0

# Sliding window approach
for right in range(N):
    # While the window size exceeds M, move the left pointer
    while A[right] - A[left] >= M:
        left += 1
    
    # Update max_gifts if current window contains more gifts
    max_gifts = max(max_gifts, right - left + 1)

# Print the result
print(max_gifts)