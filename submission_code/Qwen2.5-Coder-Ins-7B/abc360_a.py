# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Check if the plate of rice is to the left of the plate of miso soup
if S.index('R') < S.index('M'):
    print("Yes")
else:
    print("No")