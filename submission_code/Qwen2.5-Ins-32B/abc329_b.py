import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Convert the list of strings to integers
N = int(data[0])
A = list(map(int, data[1:]))

# Find the largest integer
max_A = max(A)

# Filter out the largest integer and find the new largest
second_max_A = max(filter(lambda x: x != max_A, A))

# Write the answer to stdout
print(second_max_A)