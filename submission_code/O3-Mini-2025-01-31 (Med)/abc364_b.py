def main():
    import sys
    input_data = sys.stdin.read().split()
    # Parse grid dimensions
    H = int(input_data[0])
    W = int(input_data[1])
    
    # Parse starting cell coordinates (given as 1-indexed)
    start_i = int(input_data[2])
    start_j = int(input_data[3])
    
    # Read grid rows; each row is a string
    grid = []
    index = 4
    for _ in range(H):
        grid.append(input_data[index])
        index += 1
    
    # The remaining token is the command string
    X = input_data[index]
    
    # Convert starting position to 0-index for internal processing
    r, c = start_i - 1, start_j - 1
    
    # Process each move in the command string
    for move in X:
        if move == 'L':
            new_r, new_c = r, c - 1
        elif move == 'R':
            new_r, new_c = r, c + 1
        elif move == 'U':
            new_r, new_c = r - 1, c
        elif move == 'D':
            new_r, new_c = r + 1, c
        else:
            continue  # In case of any unexpected character
        
        # Check whether the new coordinates are within bounds and the cell is empty
        if 0 <= new_r < H and 0 <= new_c < W and grid[new_r][new_c] == '.':
            r, c = new_r, new_c
    
    # Output the final position converting from 0-index to 1-index
    sys.stdout.write(f"{r + 1} {c + 1}")

if __name__ == "__main__":
    main()