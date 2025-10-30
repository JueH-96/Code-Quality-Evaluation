# YOUR CODE HERE
import sys

N = int(sys.stdin.read().strip())

result = ""
for i in range(1, N + 1):
    if i % 3 == 0:
        result += "x"
    else:
        result += "o"

print(result)