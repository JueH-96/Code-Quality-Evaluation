def dist(x1, y1, x2, y2):
    # Check if reachable: same parity for x+y and x-y
    if (x1 + y1) % 2 != (x2 + y2) % 2 or (x1 - y1) % 2 != (x2 - y2) % 2:
        return 0
    
    dx = x2 - x1
    dy = y2 - y1
    p = (dx + dy) // 2
    q = (dx - dy) // 2
    
    return abs(p) + abs(q)

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total = 0
for i in range(n):
    for j in range(i + 1, n):
        x1, y1 = points[i]
        x2, y2 = points[j]
        total += dist(x1, y1, x2, y2)

print(total)