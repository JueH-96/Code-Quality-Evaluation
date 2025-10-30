import sys

# Read the inputs from stdin
inputs = [int(i) for i in sys.stdin.read().split()]

# Calculate the result
result = sum([inputs[i] * 2**i for i in range(64)])

# Write the result to stdout
print(result)