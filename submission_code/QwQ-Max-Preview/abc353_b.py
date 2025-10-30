from collections import deque

n, k = map(int, input().split())
a = list(map(int, input().split()))
q = deque(a)

current_seats = k
count = 0

while q:
    if current_seats < q[0]:
        count += 1
        current_seats = k
    else:
        current_seats -= q.popleft()

# After processing all groups, start the attraction once
count += 1

print(count)