from collections import deque

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    queue = deque(a)
    empty_seats = k
    starts = 0

    while queue:
        if empty_seats < queue[0]:
            starts += 1
            empty_seats = k
        else:
            empty_seats -= queue.popleft()

    if n > 0:
        starts += 1

    print(starts)

solve()