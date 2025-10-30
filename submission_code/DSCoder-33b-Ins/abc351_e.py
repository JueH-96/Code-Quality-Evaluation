# YOUR CODE HERE
import sys

def calculate_distance(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if dx < dy:
        return dy
    else:
        return dx

def main():
    n = int(sys.stdin.readline())
    points = []
    for _ in range(n):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))

    total_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += calculate_distance(points[i][0], points[i][1], points[j][0], points[j][1])

    print(total_distance)

if __name__ == "__main__":
    main()