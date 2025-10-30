def min_toll(sx, sy, tx, ty):
    """
    Calculate the minimum toll Takahashi must pay to reach the point (tx+0.5, ty+0.5) from (sx+0.5, sy+0.5).
    
    Args:
        sx (int): The x-coordinate of the starting point.
        sy (int): The y-coordinate of the starting point.
        tx (int): The x-coordinate of the target point.
        ty (int): The y-coordinate of the target point.
    
    Returns:
        int: The minimum toll Takahashi must pay.
    """
    # Calculate the absolute difference in x and y coordinates
    dx = abs(tx - sx)
    dy = abs(ty - sy)
    
    # If the difference in x and y coordinates is the same, no toll is paid
    if dx == dy:
        return 0
    
    # Calculate the minimum toll
    toll = (dx + dy + 1) // 2
    
    return toll

# Read the input from stdin
sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

# Calculate and print the minimum toll
print(min_toll(sx, sy, tx, ty))