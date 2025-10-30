# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse the input
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Create a set of elements in A for quick lookup
A_set = set(A)

# Calculate the sum of integers between 1 and K that do not appear in A
result = sum(i for i in range(1, K + 1) if i not in A_set)

# Print the result
print(result)