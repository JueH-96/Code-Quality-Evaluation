# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read().split()
index = 0

N = int(input[index])
Q = int(input[index + 1])
index += 2

colors = list(map(int, input[index:index + N]))
index += N

boxes = [set() for _ in range(N + 1)]
for i in range(N):
    boxes[i + 1].add(colors[i])

result = []

for _ in range(Q):
    a = int(input[index])
    b = int(input[index + 1])
    index += 2
    
    boxes[b].update(boxes[a])
    boxes[a].clear()
    
    result.append(len(boxes[b]))

print('
'.join(map(str, result)))