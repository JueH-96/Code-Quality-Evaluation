# YOUR CODE HERE
n, d = map(int, input().split())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

# Find bounds for search space
min_x = min(x for x, y in points) - d
max_x = max(x for x, y in points) + d
min_y = min(y for x, y in points) - d
max_y = max(y for x, y in points) + d

count = 0
for x in range(min_x, max_x + 1):
    for y in range(min_y, max_y + 1):
        total_distance = 0
        for px, py in points:
            total_distance += abs(x - px) + abs(y - py)
            if total_distance > d:  # Early termination
                break
        
        if total_distance <= d:
            count += 1

print(count)