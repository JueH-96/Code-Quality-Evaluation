# YOUR CODE HERE
import sys

# Read all input lines
input_lines = sys.stdin.read().strip().split()

# Convert input lines to integers
numbers = list(map(int, input_lines))

# Reverse the list
numbers.reverse()

# Print each number in the reversed list
for number in numbers:
    print(number)