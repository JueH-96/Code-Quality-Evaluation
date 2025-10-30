def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    X = int(data[2]) - 1  # Convert to 0-based index
    Y = int(data[3]) - 1  # Convert to 0-based index
    
    grid = []
    index = 4
    for _ in range(H):
        grid.append(data[index])
        index += 1
    
    T = data[index]
    
    # Directions mapping
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }
    
    # Set to track visited houses
    visited_houses = set()
    
    # Current position of Santa
    x, y = X, Y
    
    # Process each move in T
    for move in T:
        dx, dy = directions[move]
        new_x, new_y = x + dx, y + dy
        
        # Check if the new position is within bounds and passable
        if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
            x, y = new_x, new_y
        
        # Check if the current position has a house
        if grid[x][y] == '@':
            visited_houses.add((x, y))
    
    # Output the final position (1-based) and the count of distinct houses visited
    print(x + 1, y + 1, len(visited_houses))