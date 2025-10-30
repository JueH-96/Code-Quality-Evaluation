# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    X = int(data[2])
    Y = int(data[3])
    
    grid = []
    index = 4
    for i in range(H):
        grid.append(data[index])
        index += 1
    
    T = data[index]
    
    # Adjust X and Y to be 0-indexed
    X -= 1
    Y -= 1
    
    # Set of visited houses
    visited_houses = set()
    
    # Directions mapping
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }
    
    # Function to check if a cell is passable
    def is_passable(x, y):
        if 0 <= x < H and 0 <= y < W:
            return grid[x][y] != '#'
        return False
    
    # Initial position
    current_x, current_y = X, Y
    
    # Process each move in T
    for move in T:
        dx, dy = directions[move]
        new_x, new_y = current_x + dx, current_y + dy
        
        if is_passable(new_x, new_y):
            current_x, current_y = new_x, new_y
        
        # Check if the current cell is a house
        if grid[current_x][current_y] == '@':
            visited_houses.add((current_x, current_y))
    
    # Convert back to 1-indexed for the output
    final_x = current_x + 1
    final_y = current_y + 1
    distinct_houses = len(visited_houses)
    
    print(final_x, final_y, distinct_houses)

if __name__ == "__main__":
    main()