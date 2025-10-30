# YOUR CODE HERE
import sys

# Read input from stdin
N = int(sys.stdin.read().strip())

# Calculate the number of trailing zeros in the binary representation of N
ctz = 0
while N % 2 == 0:
    N //= 2
    ctz += 1

# Print the result
print(ctz)