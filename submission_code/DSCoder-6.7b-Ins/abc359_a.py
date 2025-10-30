# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
S = [sys.stdin.readline().strip() for _ in range(N)]

count = S.count('Takahashi')

print(count)