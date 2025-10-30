import bisect

# Read inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the gift coordinates
A.sort()

# Initialize the maximum number of gifts that can be acquired
max_gifts = 0

# Iterate over the sorted gift coordinates
for i in range(N):
    # Find the rightmost position where a gift can be acquired
    rightmost = bisect.bisect_left(A, A[i] + M)
    # Update the maximum number of gifts if the current interval contains more gifts
    max_gifts = max(max_gifts, rightmost - i)

# Print the maximum number of gifts that can be acquired
print(max_gifts)