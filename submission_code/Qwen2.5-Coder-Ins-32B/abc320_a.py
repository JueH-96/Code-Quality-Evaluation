# YOUR CODE HERE
import sys

input = sys.stdin.read().strip().split()
A = int(input[0])
B = int(input[1])

result = A**B + B**A
print(result)