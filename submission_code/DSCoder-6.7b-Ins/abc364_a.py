# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
dishes = [sys.stdin.readline().strip() for _ in range(N)]

sick = False
for i in range(N-1):
    if dishes[i] == 'sweet' and dishes[i+1] == 'sweet':
        sick = True
        break

print('Yes' if not sick else 'No')