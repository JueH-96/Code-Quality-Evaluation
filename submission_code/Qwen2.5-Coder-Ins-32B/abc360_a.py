# YOUR CODE HERE
import sys

input = sys.stdin.read().strip()
if input.index('R') < input.index('M'):
    print("Yes")
else:
    print("No")