# YOUR CODE HERE
import sys

N = int(sys.stdin.read().strip())
result = ''.join('x' if i % 3 == 0 else 'o' for i in range(1, N + 1))
print(result)