def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    # Check if same parity
    if (x1 + y1) % 2 != (x2 + y2) % 2:
        return 0
    
    # If same parity, return the maximum of absolute differences
    return max(abs(x2 - x1), abs(y2 - y1))

# Read input
n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

# Calculate sum of all distances
total = 0
for i in range(n - 1):
    for j in range(i + 1, n):
        total += dist(points[i], points[j])

print(total)