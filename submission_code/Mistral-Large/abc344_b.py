import sys

# Read all input lines
input_lines = sys.stdin.read().strip().split('
')

# Convert input lines to integers
numbers = [int(line) for line in input_lines]

# Print the numbers in reverse order
for number in reversed(numbers):
    print(number)