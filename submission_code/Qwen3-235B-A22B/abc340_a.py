# Read the input values
A, B, D = map(int, input().split())

# Calculate the number of terms in the sequence
n = ((B - A) // D) + 1

# Generate the sequence
sequence = [str(A + i * D) for i in range(n)]

# Print the result
print(' '.join(sequence))