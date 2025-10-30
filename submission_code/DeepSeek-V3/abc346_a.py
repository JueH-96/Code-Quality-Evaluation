# YOUR CODE HERE
# Read input values
N, *rest = map(int, open(0).read().split())
A = rest[:N]

# Calculate B_i
B = [A[i] * A[i+1] for i in range(N-1)]

# Print the result
print(' '.join(map(str, B)))