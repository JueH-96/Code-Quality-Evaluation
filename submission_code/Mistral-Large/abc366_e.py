import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    D = int(data[index])
    index += 1

    points = []
    for _ in range(N):
        x = int(data[index])
        index += 1
        y = int(data[index])
        index += 1
        points.append((x, y))

    x_coords = sorted([point[0] for point in points])
    y_coords = sorted([point[1] for point in points])

    def count_valid_points(coords, D):
        n = len(coords)
        left_cost = [0] * n
        right_cost = [0] * n

        for i in range(1, n):
            left_cost[i] = left_cost[i-1] + (coords[i] - coords[i-1]) * i

        for i in range(n-2, -1, -1):
            right_cost[i] = right_cost[i+1] + (coords[i+1] - coords[i]) * (n-1-i)

        total_cost = left_cost[-1] + right_cost[0]
        valid_points = set()

        for i in range(n):
            current_cost = left_cost[i] + right_cost[i]
            remaining_cost = D - (total_cost - current_cost)
            if remaining_cost >= 0:
                valid_points.add(coords[i])
                if i < n-1:
                    next_point = coords[i] + remaining_cost // n
                    valid_points.add(next_point)
                if i > 0:
                    prev_point = coords[i] - remaining_cost // n
                    valid_points.add(prev_point)

        return len(valid_points)

    valid_x_count = count_valid_points(x_coords, D)
    valid_y_count = count_valid_points(y_coords, D)

    print(valid_x_count * valid_y_count)

if __name__ == "__main__":
    solve()