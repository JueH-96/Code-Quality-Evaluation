import math

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def find_farthest_points(points):
    N = len(points)
    farthest_points = []

    for i in range(N):
        max_distance = 0
        farthest_id = -1
        for j in range(N):
            if i != j:
                distance = euclidean_distance(points[i][0], points[i][1], points[j][0], points[j][1])
                if distance > max_distance:
                    max_distance = distance
                    farthest_id = j
                elif distance == max_distance:
                    farthest_id = min(farthest_id, j)
        farthest_points.append(farthest_id + 1)

    return farthest_points

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    points = []
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2

    farthest_points = find_farthest_points(points)
    for point in farthest_points:
        print(point)

if __name__ == "__main__":
    main()