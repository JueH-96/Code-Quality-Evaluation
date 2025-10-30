# YOUR CODE HERE
n, d = map(int, input().split())
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

count = 0
for x in range(-1000, 1001):
    for y in range(-1000, 1001):
        dist_sum = 0
        for px, py in points:
            dist_sum += abs(x - px) + abs(y - py)
        if dist_sum <= d:
            count += 1

print(count)