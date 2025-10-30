N, M, H, K = map(int, input().split())
S = input()
points = [tuple(map(int, input().split())) for _ in range(M)]

x, y, health = 0, 0, H
for s in S:
    if s == 'R':
        x += 1
    elif s == 'L':
        x -= 1
    elif s == 'U':
        y += 1
    elif s == 'D':
        y -= 1
    health -= 1
    if health < 0:
        break
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if (nx, ny) in points:
            health = max(health, nx)
            break

print('Yes' if health >= 0 else 'No')