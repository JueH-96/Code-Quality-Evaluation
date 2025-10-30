def calculate_distance(x1, y1, x2, y2):
    # Check if the points are reachable
    if (x1 + y1) % 2 != (x2 + y2) % 2:
        return 0  # Not reachable
    # Calculate the distance
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max((dx + dy) // 2, (dx + dy + 1) // 2)

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N = int(data[0])
    points = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    total_distance = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            total_distance += calculate_distance(points[i][0], points[i][1], points[j][0], points[j][1])
    
    print(total_distance)

if __name__ == "__main__":
    main()