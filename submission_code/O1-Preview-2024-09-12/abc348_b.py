# YOUR CODE HERE
N = int(input())
points = []
for i in range(N):
    x_str, y_str = input().split()
    x = int(x_str)
    y = int(y_str)
    points.append((x,y))

for i in range(N):
    max_dist = -1
    max_ids = []
    xi, yi = points[i]
    for j in range(N):
        if i == j:
            continue
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_dist:
            max_dist = dist
            max_ids = [j+1]  # IDs from 1 to N
        elif dist == max_dist:
            max_ids.append(j+1)
    farthest_id = min(max_ids)
    print(farthest_id)