def main():
    import sys

    data = sys.stdin.read().strip().split()
    H = int(data[0])
    W = int(data[1])
    S_i = int(data[2])
    S_j = int(data[3])
    
    grid = data[4:4+H]
    X = data[4+H]
    
    # Convert to 0-based indexing
    current_row = S_i - 1
    current_col = S_j - 1
    
    for move in X:
        new_row, new_col = current_row, current_col
        
        if move == 'L':
            new_col -= 1
        elif move == 'R':
            new_col += 1
        elif move == 'U':
            new_row -= 1
        elif move == 'D':
            new_row += 1
        
        # Check bounds and whether the cell is empty
        if 0 <= new_row < H and 0 <= new_col < W and grid[new_row][new_col] == '.':
            current_row, current_col = new_row, new_col
    
    # Convert back to 1-based indexing
    print(current_row + 1, current_col + 1)

# Do not remove the call to main().
main()