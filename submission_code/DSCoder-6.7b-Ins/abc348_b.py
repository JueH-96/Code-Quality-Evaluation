import sys
import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def find_farthest_point(points, origin):
    max_distance = -1
    farthest_point = None
    for point in points:
        if point != origin:
            distance = calculate_distance(origin[0], origin[1], point[0], point[1])
            if distance > max_distance:
                max_distance = distance
                farthest_point = point
    return farthest_point

def main():
    N = int(sys.stdin.readline().strip())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().strip().split())
        points.append((x, y))
    for i in range(1, N+1):
        origin = points[i-1]
        farthest_point = find_farthest_point(points, origin)
        print(points.index(farthest_point) + 1)

if __name__ == "__main__":
    main()