# YOUR CODE HERE
import sys

# Read all input lines
input_lines = sys.stdin.read().strip().split()

# Initialize the count of valid strings
count = 0

# Iterate over each string and its index
for i in range(12):
    if len(input_lines[i]) == i + 1:
        count += 1

# Print the result
print(count)