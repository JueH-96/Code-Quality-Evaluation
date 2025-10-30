# YOUR CODE HERE
from collections import defaultdict

N, Q = map(int, input().split())
C = list(map(int, input().split()))

boxes = defaultdict(set)
for i in range(N):
    boxes[C[i]].add(i+1)

for _ in range(Q):
    a, b = map(int, input().split())
    boxes[C[b-1]].update(boxes[C[a-1]])
    boxes[C[a-1]].clear()
    C[b-1] = C[a-1]
    print(len(boxes[C[b-1]]))