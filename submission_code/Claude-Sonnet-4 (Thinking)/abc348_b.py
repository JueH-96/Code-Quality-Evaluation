n = int(input())
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

for i in range(n):
    max_distance_sq = -1
    farthest_point_id = -1
    
    for j in range(n):
        if i == j:
            continue
        
        x1, y1 = points[i]
        x2, y2 = points[j]
        distance_sq = (x1 - x2)**2 + (y1 - y2)**2
        
        if distance_sq > max_distance_sq or (distance_sq == max_distance_sq and j + 1 < farthest_point_id):
            max_distance_sq = distance_sq
            farthest_point_id = j + 1
    
    print(farthest_point_id)