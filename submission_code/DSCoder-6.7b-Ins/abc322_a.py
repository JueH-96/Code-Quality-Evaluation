# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

result = -1

for i in range(N-2):
    if S[i:i+3] == 'ABC':
        result = i+1
        break

print(result)