# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
A = int(data[0])
B = int(data[1])
result = A**B + B**A
print(result)