import sys

n, m, h, k = map(int, sys.stdin.readline().split())
s = sys.stdin.readline().strip()
items = set()

for _ in range(m):
    x, y = map(int, sys.stdin.readline().split())
    items.add((x, y))

x, y = 0, 0
current_health = h

for c in s:
    if c == 'R':
        x += 1
    elif c == 'L':
        x -= 1
    elif c == 'U':
        y += 1
    elif c == 'D':
        y -= 1

    current_health -= 1

    if current_health < 0:
        print("No")
        exit()

    if (x, y) in items:
        if current_health < k:
            current_health = k
            items.remove((x, y))

print("Yes")