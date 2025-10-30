import sys

# Read input from stdin
input = sys.stdin.read()

# Split the input into B and G
B, G = map(int, input.split())

# Determine which item is more expensive and print the result
if B > G:
    print("Bat")
else:
    print("Glove")