# YOUR CODE HERE
import sys

X = float(sys.stdin.read().strip())
print(f"{X:.3f}".rstrip('0').rstrip('.'))