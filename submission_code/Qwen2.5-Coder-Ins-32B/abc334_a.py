# YOUR CODE HERE
import sys

B, G = map(int, sys.stdin.read().split())
if B > G:
    print("Bat")
else:
    print("Glove")