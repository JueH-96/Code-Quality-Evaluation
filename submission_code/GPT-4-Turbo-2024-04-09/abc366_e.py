import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    points = [(int(data[i*2+2]), int(data[i*2+3])) for i in range(N)]
    
    # To find the bounds for x and y where the Manhattan distance sum could be <= D
    # We need to consider the extreme cases where the sum of distances is minimized
    # We will use a bounding box approach based on the Manhattan distance properties
    
    # Calculate the bounding box that satisfies the Manhattan distance condition
    # We use the fact that for a point (x, y) to satisfy the condition:
    # sum(|x - xi| + |y - yi|) <= D
    # We can derive bounds for x and y independently
    
    # For each point (xi, yi), the condition |x - xi| + |y - yi| <= D
    # can be transformed into:
    # -D <= x - xi + y - yi <= D
    # -D <= x - xi - (y - yi) <= D
    # which gives us:
    # x + y <= xi + yi + D
    # x + y >= xi + yi - D
    # x - y <= xi - yi + D
    # x - y >= xi - yi - D
    
    # We need to find the intersection of all these bounds for all points
    max_x_plus_y = float('-inf')
    min_x_plus_y = float('inf')
    max_x_minus_y = float('-inf')
    min_x_minus_y = float('inf')
    
    for x, y in points:
        max_x_plus_y = max(max_x_plus_y, x + y + D)
        min_x_plus_y = min(min_x_plus_y, x + y - D)
        max_x_minus_y = max(max_x_minus_y, x - y + D)
        min_x_minus_y = min(min_x_minus_y, x - y - D)
    
    # Now we need to count all integer pairs (x, y) that satisfy:
    # min_x_plus_y <= x + y <= max_x_plus_y
    # min_x_minus_y <= x - y <= max_x_minus_y
    
    # To count valid (x, y), we iterate over possible x + y and x - y values
    # and derive possible x and y from them.
    
    count = 0
    
    # We iterate over all possible values of x + y and x - y within the bounds
    for x_plus_y in range(min_x_plus_y, max_x_plus_y + 1):
        for x_minus_y in range(min_x_minus_y, max_x_minus_y + 1):
            # From x + y = x_plus_y and x - y = x_minus_y
            # We can derive x and y:
            # x = (x_plus_y + x_minus_y) / 2
            # y = (x_plus_y - x_minus_y) / 2
            # These must be integers
            if (x_plus_y + x_minus_y) % 2 == 0 and (x_plus_y - x_minus_y) % 2 == 0:
                x = (x_plus_y + x_minus_y) // 2
                y = (x_plus_y - x_minus_y) // 2
                # Check if this (x, y) satisfies the Manhattan distance condition
                if all(abs(x - xi) + abs(y - yi) <= D for xi, yi in points):
                    count += 1
    
    print(count)

if __name__ == "__main__":
    main()