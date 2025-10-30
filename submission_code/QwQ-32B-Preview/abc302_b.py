import sys

def main():
    # Read H and W
    H_W = sys.stdin.readline().strip()
    while H_W == '':
        H_W = sys.stdin.readline().strip()
    H, W = map(int, H_W.split())
    
    # Read the grid
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        while line == '':
            line = sys.stdin.readline().strip()
        grid.append(list(line))
    
    # Define the sequence to find
    sequence = ['s', 'n', 'u', 'k', 'e']
    
    # Define the eight possible directions
    directions = [
        (0, 1),   # East
        (0, -1),  # West
        (-1, 0),  # North
        (1, 0),   # South
        (-1, 1),  # North-East
        (-1, -1), # North-West
        (1, 1),   # South-East
        (1, -1)   # South-West
    ]
    
    # Iterate through each cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                # Check in each direction
                for direction in directions:
                    positions = [(i, j)]
                    current_i, current_j = i, j
                    match = True
                    for letter in sequence[1:]:
                        current_i += direction[0]
                        current_j += direction[1]
                        if 0 <= current_i < H and 0 <= current_j < W and grid[current_i][current_j] == letter:
                            positions.append((current_i, current_j))
                        else:
                            match = False
                            break
                    if match:
                        # Convert to 1-indexed positions
                        positions_1indexed = [(p[0]+1, p[1]+1) for p in positions]
                        # Print the positions
                        for pos in positions_1indexed:
                            print(pos[0], pos[1])
                        sys.exit()

if __name__ == "__main__":
    main()