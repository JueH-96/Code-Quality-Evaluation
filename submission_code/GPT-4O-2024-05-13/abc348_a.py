# YOUR CODE HERE
import sys
input = sys.stdin.read

N = int(input().strip())

result = []
for i in range(1, N + 1):
    if i % 3 == 0:
        result.append('x')
    else:
        result.append('o')

print(''.join(result))