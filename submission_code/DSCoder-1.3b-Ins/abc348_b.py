import sys
import math

def read_input():
    N = int(sys.stdin.readline().strip())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().strip().split())
        points.append((x, y))
    return N, points

def find_farthest_point(N, points):
    farthest_point_id = None
    max_distance = None
    for i in range(N):
        for j in range(i+1, N):
            point_i, point_j = points[i], points[j]
            distance = math.sqrt((point_i[0] - point_j[0])**2 + (point_i[1] - point_j[1])**2)
            if max_distance is None or distance > max_distance:
                max_distance = distance
                farthest_point_id = min(point_i, point_j)
    return farthest_point_id[0]

def main():
    N, points = read_input()
    farthest_point_id = find_farthest_point(N, points)
    print(farthest_point_id)

if __name__ == "__main__":
    main()