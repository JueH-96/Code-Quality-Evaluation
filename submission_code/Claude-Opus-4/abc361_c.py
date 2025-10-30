# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Sort the array
A.sort()

# We need to keep N-K elements
keep = N - K

# Find minimum difference among all possible windows of size 'keep'
min_diff = float('inf')
for i in range(N - keep + 1):
    # Window from i to i+keep-1
    diff = A[i + keep - 1] - A[i]
    min_diff = min(min_diff, diff)

print(min_diff)