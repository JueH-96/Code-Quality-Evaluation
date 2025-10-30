def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line for H, W, X, Y
    H, W, X, Y = map(int, data[0].split())
    
    # Read the grid
    grid = [list(data[i + 1]) for i in range(H)]
    
    # Read the movement string T
    T = data[H + 1]
    
    # Adjust X and Y to be zero-indexed
    X -= 1
    Y -= 1
    
    # Directions mapping
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }
    
    # Set to track distinct houses
    visited_houses = set()
    
    # Start at the initial position
    current_x, current_y = X, Y
    
    # Check if the starting position is a house
    if grid[current_x][current_y] == '@':
        visited_houses.add((current_x, current_y))
    
    # Process each move in T
    for move in T:
        dx, dy = directions[move]
        new_x, new_y = current_x + dx, current_y + dy
        
        # Check bounds and if the new cell is passable
        if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
            current_x, current_y = new_x, new_y
            
            # Check if the new position is a house
            if grid[current_x][current_y] == '@':
                visited_houses.add((current_x, current_y))
    
    # Resulting position and count of distinct houses
    result_x = current_x + 1  # Convert back to 1-indexed
    result_y = current_y + 1  # Convert back to 1-indexed
    count_houses = len(visited_houses)
    
    # Print the result
    print(result_x, result_y, count_houses)

if __name__ == "__main__":
    main()