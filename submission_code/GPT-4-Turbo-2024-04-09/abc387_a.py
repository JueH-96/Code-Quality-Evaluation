# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().strip()
A, B = map(int, data.split())
result = (A + B) ** 2
print(result)