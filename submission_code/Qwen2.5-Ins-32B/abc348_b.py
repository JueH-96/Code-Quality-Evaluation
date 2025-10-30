import sys
import math

def find_farthest_points(n, points):
    distances = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                distances[i][j] = dist

    result = []
    for i in range(n):
        max_dist = max(distances[i])
        farthest_points = [j for j, dist in enumerate(distances[i]) if dist == max_dist]
        result.append(min(farthest_points) + 1)
    return result

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    n = int(data[index])
    index += 1
    points = []
    for i in range(n):
        x = int(data[index])
        index += 1
        y = int(data[index])
        index += 1
        points.append((x, y))
    
    result = find_farthest_points(n, points)
    for r in result:
        print(r)