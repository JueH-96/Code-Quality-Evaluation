# Read the first line containing N, K, X
N, K, X = map(int, input().split())

# Read the second line containing the array A
A = list(map(int, input().split()))

# Create array B by inserting X after the K-th element
# Note: Since array is 1-indexed in the problem, we insert after position K
B = A[:K] + [X] + A[K:]

# Print the resulting array B
print(' '.join(map(str, B)))