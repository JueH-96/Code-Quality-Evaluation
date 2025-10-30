# Read the input line
input_line = input()

# Split the input into parts and convert to integers
parts = input_line.split()
N = int(parts[0])
L = int(parts[1])
R = int(parts[2])

# Create the initial sequence A
A = list(range(1, N+1))

# Reverse the sublist from A[L-1] to A[R-1] (R is inclusive, so we use R)
A[L-1:R] = A[L-1:R][::-1]

# Print the modified sequence
print(' '.join(map(str, A)))