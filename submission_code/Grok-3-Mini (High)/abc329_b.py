import sys  # Not necessary, but can be used; however, using built-in input() is fine.

# Read N from input
N = int(input())

# Read the list of integers from input and convert to integers
A = list(map(int, input().split()))

# Find the maximum value in the list
max_val = max(A)

# Find the maximum value that is less than the maximum value
second_max = max(num for num in A if num < max_val)

# Print the result
print(second_max)