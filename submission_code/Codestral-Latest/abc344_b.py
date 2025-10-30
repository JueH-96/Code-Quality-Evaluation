# YOUR CODE HERE
import sys

# Read all input lines
lines = sys.stdin.read().strip().split()

# Convert the input lines to integers
numbers = list(map(int, lines))

# Reverse the list
numbers.reverse()

# Print the reversed list
for number in numbers:
    print(number)