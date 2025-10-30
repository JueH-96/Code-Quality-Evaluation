# YOUR CODE HERE
import sys

N = int(sys.stdin.read().strip())

result = []
for i in range(N + 1):
    found = False
    for j in range(1, 10):
        if N % j == 0 and i % (N // j) == 0:
            result.append(str(j))
            found = True
            break
    if not found:
        result.append('-')

print(''.join(result))