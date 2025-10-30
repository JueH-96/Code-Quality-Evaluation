# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Initialize a counter for the number of valid strings
count = 0

# Iterate over the first 12 strings
for i in range(12):
    # Check if the length of the string is equal to its index (1-based)
    if len(data[i]) == i + 1:
        count += 1

# Print the result
print(count)