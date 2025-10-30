# YOUR CODE HERE
# Read input values
N, L, R = map(int, input().split())

# Initialize the sequence A
A = list(range(1, N+1))

# Adjust L and R to 0-based indexing
L -= 1
R -= 1

# Reverse the sublist from L to R
A[L:R+1] = A[L:R+1][::-1]

# Print the result
print(' '.join(map(str, A)))