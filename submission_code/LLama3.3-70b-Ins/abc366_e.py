import sys

def calculate_distance(x, y, points):
    """
    Calculate the sum of Manhattan distances from (x, y) to all points.
    
    Args:
    x (int): The x-coordinate of the point.
    y (int): The y-coordinate of the point.
    points (list): A list of points.
    
    Returns:
    int: The sum of Manhattan distances.
    """
    return sum(abs(x - point[0]) + abs(y - point[1]) for point in points)

def solve(N, D, points):
    """
    Solve the problem.
    
    Args:
    N (int): The number of points.
    D (int): The maximum sum of Manhattan distances.
    points (list): A list of points.
    
    Returns:
    int: The number of integer pairs (x, y) that satisfy the condition.
    """
    count = 0
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    
    for x in range(min_x - D, max_x + D + 1):
        for y in range(min_y - D, max_y + D + 1):
            if calculate_distance(x, y, points) <= D:
                count += 1
                
    return count

def main():
    """
    Read the input and solve the problem.
    """
    N, D = map(int, sys.stdin.readline().split())
    points = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    print(solve(N, D, points))

if __name__ == "__main__":
    main()