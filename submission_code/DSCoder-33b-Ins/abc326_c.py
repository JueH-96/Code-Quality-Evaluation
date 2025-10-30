# YOUR CODE HERE

from collections import deque

N, M = map(int, input().split())
A = list(map(int, input().split()))

A.sort()

queue = deque()
max_gifts = 0

for a in A:
    while queue and queue[0] < a - M:
        queue.popleft()
    queue.append(a)
    max_gifts = max(max_gifts, len(queue))

print(max_gifts)