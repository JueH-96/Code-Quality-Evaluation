# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the gift positions
A.sort()

max_gifts = 0

# Use two pointers approach
left = 0
for right in range(N):
    # Move left pointer while the interval is too large
    while left < right and A[right] - A[left] >= M:
        left += 1
    
    # Count gifts in interval [A[left], A[left] + M)
    # All gifts from index left to right are in this interval
    max_gifts = max(max_gifts, right - left + 1)

print(max_gifts)