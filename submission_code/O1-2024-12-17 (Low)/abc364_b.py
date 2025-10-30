def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    
    H, W = map(int, data[:2])
    S_i, S_j = map(int, data[2:4])
    
    grid_index = 4
    grid = []
    
    for row in range(H):
        grid.append(data[grid_index])
        grid_index += 1
    
    X = data[grid_index]
    
    # Current position (row, column) - using 1-based indexing
    cur_row, cur_col = S_i, S_j
    
    for move in X:
        if move == 'L':
            next_row, next_col = cur_row, cur_col - 1
        elif move == 'R':
            next_row, next_col = cur_row, cur_col + 1
        elif move == 'U':
            next_row, next_col = cur_row - 1, cur_col
        elif move == 'D':
            next_row, next_col = cur_row + 1, cur_col
        
        # Check boundaries
        if 1 <= next_row <= H and 1 <= next_col <= W:
            # Check if the next cell is empty
            if grid[next_row - 1][next_col - 1] == '.':
                cur_row, cur_col = next_row, next_col
    
    print(cur_row, cur_col)

# Do not forget to call the main function
if __name__ == "__main__":
    main()