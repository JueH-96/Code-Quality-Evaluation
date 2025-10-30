import math
import sys

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_farthest_point(points, point_id):
    """Find the farthest point from a given point."""
    max_distance = 0
    farthest_point_id = None
    for i, point in enumerate(points):
        if i != point_id:
            distance = calculate_distance(points[point_id], point)
            if distance > max_distance:
                max_distance = distance
                farthest_point_id = i + 1
            elif distance == max_distance:
                farthest_point_id = min(farthest_point_id, i + 1)
    return farthest_point_id

def main():
    """Read input and print the farthest point for each point."""
    n = int(input())
    points = [tuple(map(int, input().split())) for _ in range(n)]
    for i in range(n):
        farthest_point_id = find_farthest_point(points, i)
        print(farthest_point_id)

if __name__ == "__main__":
    main()