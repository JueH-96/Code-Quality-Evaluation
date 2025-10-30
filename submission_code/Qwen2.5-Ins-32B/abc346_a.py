import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Convert the first element to integer to get N
N = int(data[0])

# Convert the remaining elements to integers and store them in a list
A = list(map(int, data[1:]))

# Calculate B_i = A_i * A_{i+1} for 1 <= i <= N-1
B = [A[i] * A[i+1] for i in range(N-1)]

# Print the result
print(' '.join(map(str, B)))