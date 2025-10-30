# YOUR CODE HERE
def max_gifts_in_interval(N, M, A):
    # Sort the gift coordinates
    A.sort()
    
    # Initialize two pointers
    left = 0
    max_gifts = 0
    
    # Use the right pointer to expand the interval
    for right in range(N):
        # Check if the current interval is valid
        while A[right] - A[left] >= M:
            left += 1
        # Update the maximum number of gifts in a valid interval
        max_gifts = max(max_gifts, right - left + 1)
    
    return max_gifts

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

print(max_gifts_in_interval(N, M, A))