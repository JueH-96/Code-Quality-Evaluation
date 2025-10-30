import sys

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the minimum number of jumps required to get from point (x1, y1) to point (x2, y2).
    
    Args:
    x1 (int): The x-coordinate of the starting point.
    y1 (int): The y-coordinate of the starting point.
    x2 (int): The x-coordinate of the ending point.
    y2 (int): The y-coordinate of the ending point.
    
    Returns:
    int: The minimum number of jumps required to get from point (x1, y1) to point (x2, y2).
    """
    # Calculate the absolute difference in x and y coordinates
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # If the difference in x and y coordinates is not the same, it's impossible to reach the point
    if (dx + dy) % 2 != 0:
        return 0
    
    # Calculate the minimum number of jumps required
    return (dx + dy) // 2

def main():
    # Read the number of points
    N = int(sys.stdin.readline())
    
    # Read the coordinates of each point
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))
    
    # Initialize the total distance
    total_distance = 0
    
    # Calculate the distance between each pair of points
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1 = points[i]
            x2, y2 = points[j]
            distance = calculate_distance(x1, y1, x2, y2)
            total_distance += distance
    
    # Print the total distance
    print(total_distance)

if __name__ == "__main__":
    main()