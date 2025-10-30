# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Sort the array
A.sort()

# We need to keep N-K elements
keep = N - K

# Find the minimum range among all windows of size 'keep'
min_range = float('inf')
for i in range(N - keep + 1):
    current_range = A[i + keep - 1] - A[i]
    min_range = min(min_range, current_range)

print(min_range)