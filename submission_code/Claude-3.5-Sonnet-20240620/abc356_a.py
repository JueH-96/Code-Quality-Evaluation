# YOUR CODE HERE
N, L, R = map(int, input().split())

# Create the initial sequence
sequence = list(range(1, N + 1))

# Reverse the subsequence from L to R
sequence[L-1:R] = reversed(sequence[L-1:R])

# Print the result
print(*sequence)