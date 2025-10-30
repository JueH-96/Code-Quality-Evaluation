import math

def solve():
    n = int(input())
    points = []
    for i in range(n):
        x, y = map(int, input().split())
        points.append((x, y, i + 1))
    
    for i in range(n):
        max_dist = -1
        farthest_point_id = -1
        for j in range(n):
            if i == j:
                continue
            x1, y1, id1 = points[i]
            x2, y2, id2 = points[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist > max_dist:
                max_dist = dist
                farthest_point_id = id2
            elif dist == max_dist:
                farthest_point_id = min(farthest_point_id, id2)
        print(farthest_point_id)

solve()