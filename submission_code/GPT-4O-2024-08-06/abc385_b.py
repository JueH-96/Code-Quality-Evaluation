# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line
    H, W, X, Y = map(int, data[0].split())
    
    # Read the grid
    grid = [list(data[i + 1]) for i in range(H)]
    
    # Read the movement string
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
    
    # Set to keep track of distinct houses
    visited_houses = set()
    
    # Process each move
    for move in T:
        dx, dy = directions[move]
        new_x, new_y = X + dx, Y + dy
        
        # Check if the new position is within bounds and passable
        if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
            X, Y = new_x, new_y
            if grid[X][Y] == '@':
                visited_houses.add((X, Y))
    
    # Output the final position and the number of distinct houses visited
    print(X + 1, Y + 1, len(visited_houses))

if __name__ == "__main__":
    main()