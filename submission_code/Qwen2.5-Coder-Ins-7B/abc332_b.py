# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

K = int(data[0])
G = int(data[1])
M = int(data[2])

glass = 0
mug = 0

for _ in range(K):
    if glass == G:
        glass = 0
    elif mug == 0:
        mug = M
    else:
        transfer = min(mug, G - glass)
        glass += transfer
        mug -= transfer

print(glass, mug)