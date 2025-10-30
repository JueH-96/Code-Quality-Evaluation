N = int(input())
points = []
for i in range(1, N+1):
    x, y = map(int, input().split())
    points.append((i, x, y))

for i in range(N):
    current_id, current_x, current_y = points[i]
    max_distance = -1.0
    farthest_id = -1
    for j in range(N):
        if i == j:
            continue
        other_id, other_x, other_y = points[j]
        distance = ((current_x - other_x)**2 + (current_y - other_y)**2)**0.5
        if distance > max_distance:
            max_distance = distance
            farthest_id = other_id
        elif abs(distance - max_distance) < 1e-9:
            if other_id < farthest_id:
                farthest_id = other_id
    print(farthest_id)