from collections import deque

n, k = map(int, input().split())
a = list(map(int, input().split()))
q = deque(a)

empty_seats = k
count = 0

while True:
    if not q:
        count += 1
        break
    front = q[0]
    if front > empty_seats:
        count += 1
        empty_seats = k
    else:
        empty_seats -= front
        q.popleft()

print(count)