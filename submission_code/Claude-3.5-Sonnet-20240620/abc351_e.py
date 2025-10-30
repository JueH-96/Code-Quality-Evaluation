# YOUR CODE HERE
def calculate_distance(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if (dx + dy) % 2 == 1:
        return 0
    return max(dx, dy)

N = int(input())
points = [tuple(map(int, input().split())) for _ in range(N)]

total_distance = 0
for i in range(N - 1):
    for j in range(i + 1, N):
        total_distance += calculate_distance(*points[i], *points[j])

print(total_distance)