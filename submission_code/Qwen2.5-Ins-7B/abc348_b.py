# N = int(input())
# points = [tuple(map(int, input().split())) for _ in range(N)]

N = 6
points = [(3, 2), (1, 6), (4, 5), (1, 3), (5, 5), (9, 8)]

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

for i in range(N):
    max_distance = 0
    farthest_point_id = i + 1
    for j in range(N):
        if i != j:
            distance = euclidean_distance(points[i], points[j])
            if distance > max_distance:
                max_distance = distance
                farthest_point_id = j + 1
    print(farthest_point_id)