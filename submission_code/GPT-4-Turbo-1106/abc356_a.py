# Read the inputs from stdin
N, L, R = map(int, input().split())

# Create the initial sequence
sequence = list(range(1, N + 1))

# Perform the reverse operation on the specified segment
sequence[L-1:R] = reversed(sequence[L-1:R])

# Print the sequence after the operation
print(' '.join(map(str, sequence)))