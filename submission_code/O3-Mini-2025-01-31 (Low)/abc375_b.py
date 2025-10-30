import math
import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    # starting point
    prev_x, prev_y = 0, 0
    total_distance = 0.0

    points = []
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))
    
    # move through the points
    for x, y in points:
        dx = x - prev_x
        dy = y - prev_y
        total_distance += math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = x, y
    
    # return to origin from last point
    total_distance += math.sqrt(prev_x * prev_x + prev_y * prev_y)
    
    # print the final result with enough precision
    print(total_distance)

if __name__ == '__main__':
    main()