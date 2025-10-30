import sys
import math

def find_farthest_point(points):
    n = len(points)
    farthest_points = [0] * n
    for i in range(n):
        max_distance = 0
        farthest_point = 0
        for j in range(n):
            if i != j:
                distance = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    farthest_point = j
        farthest_points[i] = farthest_point + 1
    return farthest_points

def main():
    n = int(sys.stdin.readline().strip())
    points = []
    for _ in range(n):
        x, y = map(int, sys.stdin.readline().strip().split())
        points.append((x, y))
    farthest_points = find_farthest_point(points)
    for point in farthest_points:
        print(point)

if __name__ == "__main__":
    main()