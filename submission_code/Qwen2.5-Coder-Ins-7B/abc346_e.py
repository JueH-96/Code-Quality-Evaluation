# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
M = int(data[2])

rows = [0] * H
cols = [0] * W

for i in range(M):
    T = int(data[3 + 3 * i])
    A = int(data[4 + 3 * i]) - 1
    X = int(data[5 + 3 * i])
    
    if T == 1:
        rows[A] = X
    elif T == 2:
        cols[A] = X

from collections import Counter

color_count = Counter()
for i in range(H):
    for j in range(W):
        color = rows[i] if rows[i] != 0 else cols[j]
        if color != 0:
            color_count[color] += 1

K = len(color_count)
print(K)
for color, count in sorted(color_count.items()):
    print(color, count)