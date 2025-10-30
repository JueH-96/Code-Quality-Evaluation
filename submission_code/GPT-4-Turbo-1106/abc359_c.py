def calculate_toll(S_x, S_y, T_x, T_y):
    # Calculate the difference in coordinates
    dx = abs(T_x - S_x)
    dy = abs(T_y - S_y)
    
    # Calculate the number of tiles crossed horizontally and vertically
    horizontal_tiles = dx // 2 + dx % 2
    vertical_tiles = dy // 2 + dy % 2
    
    # If both starting and ending points are on the same tile, no toll is paid
    if (S_x // 2 == T_x // 2) and (S_y // 2 == T_y // 2):
        return 0
    
    # If the starting and ending points are on the same row or column
    # and the row or column is even, then the toll is the number of tiles crossed
    if (S_x // 2 == T_x // 2) or (S_y // 2 == T_y // 2):
        return horizontal_tiles + vertical_tiles
    
    # Otherwise, the toll is the sum of horizontal and vertical tiles crossed minus 1
    # because the last move will not enter a new tile
    return horizontal_tiles + vertical_tiles - 1

# Read input
S_x, S_y = map(int, input().split())
T_x, T_y = map(int, input().split())

# Calculate and print the minimum toll
print(calculate_toll(S_x, S_y, T_x, T_y))