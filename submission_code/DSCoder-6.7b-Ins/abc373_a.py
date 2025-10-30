# YOUR CODE HERE

import sys

count = 0
for i in range(1, 13):
    S = sys.stdin.readline().strip()
    if len(S) == i:
        count += 1

print(count)