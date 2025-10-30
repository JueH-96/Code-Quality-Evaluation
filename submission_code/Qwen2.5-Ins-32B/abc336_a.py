import sys

# Read the input from stdin
N = int(input().strip())

# Generate the Dragon String of level N
dragon_string = "L" + "o" * N + "ng"

# Write the answer to stdout
print(dragon_string)