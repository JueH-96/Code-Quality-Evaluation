import sys

# Read all input from stdin and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to keep track of position in the data list
index = 0

# Read N from the data list
N = int(data[index])
index += 1

# Read the A sequence into a list of integers
A = list(map(int, data[index:index + N]))
index += N

# Read the B sequence into a list of integers
B = list(map(int, data[index:index + N]))

# Find the maximum values in A and B and compute their sum
max_sum = max(A) + max(B)

# Print the result
print(max_sum)