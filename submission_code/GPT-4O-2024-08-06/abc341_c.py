# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line
    H, W, N = map(int, data[0].split())
    T = data[1]
    grid = data[2:H+2]
    
    # Directions mapping
    direction_map = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0)
    }
    
    # Function to check if a position is within bounds and is land
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'
    
    # Possible end positions
    possible_end_positions = set()
    
    # Iterate over each cell in the grid
    for i in range(1, H-1):
        for j in range(1, W-1):
            if grid[i][j] == '.':
                # Start from this position
                x, y = i, j
                valid = True
                for move in T:
                    dx, dy = direction_map[move]
                    x += dx
                    y += dy
                    if not is_valid(x, y):
                        valid = False
                        break
                if valid:
                    possible_end_positions.add((x, y))
    
    # Print the number of possible end positions
    print(len(possible_end_positions))

if __name__ == "__main__":
    main()