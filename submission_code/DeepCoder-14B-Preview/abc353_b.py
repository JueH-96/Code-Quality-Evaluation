n, k = map(int, input().split())
a = list(map(int, input().split()))
from collections import deque

queue = deque(a)
seats = k
count = 0

while queue:
    current = queue[0]
    if current <= seats:
        seats -= current
        queue.popleft()
    else:
        count += 1
        seats = k

print(count)