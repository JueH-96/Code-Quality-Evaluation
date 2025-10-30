# YOUR CODE HERE
import sys

s = sys.stdin.read().strip()
s = s.rstrip('0').rstrip('.')
print(s if s else '0')