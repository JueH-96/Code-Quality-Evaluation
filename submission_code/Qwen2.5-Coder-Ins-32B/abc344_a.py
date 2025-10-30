# YOUR CODE HERE
import sys

input = sys.stdin.read().strip()
parts = input.split('|')
result = parts[0] + parts[2]
print(result)