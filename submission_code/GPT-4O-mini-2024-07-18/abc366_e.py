def count_integer_pairs(N, D, points):
    # Extract x and y coordinates from points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Determine the min and max bounds for x and y
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # The total number of valid integer pairs
    count = 0

    # Check all integer pairs (x, y) within the bounding box
    for x in range(min_x - D, max_x + D + 1):
        for y in range(min_y - D, max_y + D + 1):
            # Calculate the Manhattan distance sum
            manhattan_sum = sum(abs(x - x_i) + abs(y - y_i) for x_i, y_i in points)
            if manhattan_sum <= D:
                count += 1

    return count

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, D = map(int, data[0].split())
    points = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = count_integer_pairs(N, D, points)
    print(result)

if __name__ == "__main__":
    main()