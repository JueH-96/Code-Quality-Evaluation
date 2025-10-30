# YOUR CODE HERE
import sys

# Read input from stdin
N = int(sys.stdin.read().strip())

# Generate the Dragon String of level N
dragon_string = "L" + "o" * N + "n" + "g"

# Print the result
print(dragon_string)