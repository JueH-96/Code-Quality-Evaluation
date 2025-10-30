# YOUR CODE HERE
import sys

# Read the input
N = int(sys.stdin.read().strip())

# Construct the Dragon String of level N
dragon_string = 'L' + 'o' * N + 'ng'

# Print the result
print(dragon_string)