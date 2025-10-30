import sys
import math

input = sys.stdin.read
data = input().split()

N = int(data[0])
points = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]

def distance_squared(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

for i in range(N):
    max_distance = -1
    farthest_id = -1
    x1, y1 = points[i]
    
    for j in range(N):
        if i == j:
            continue
        x2, y2 = points[j]
        dist = distance_squared((x1, y1), (x2, y2))
        
        if dist > max_distance:
            max_distance = dist
            farthest_id = j + 1
        elif dist == max_distance:
            if j + 1 < farthest_id:
                farthest_id = j + 1
    
    print(farthest_id)