n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total = 0
for i in range(n - 1):
    for j in range(i + 1, n):
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # Check if they have same parity
        if dx % 2 == dy % 2:
            dist = max(dx, dy)
        else:
            dist = 0
        
        total += dist

print(total)