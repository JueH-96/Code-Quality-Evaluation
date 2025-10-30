def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H_W = data[0].split()
    H = int(H_W[0])
    W = int(H_W[1])
    
    S = data[1].split()
    S_i = int(S[0]) - 1  # Convert to zero-based indexing
    S_j = int(S[1]) - 1
    
    grid = []
    for i in range(H):
        row = data[2 + i].strip()
        grid.append(list(row))
    
    X = data[2 + H].strip()
    
    row = S_i
    col = S_j
    
    for dir in X:
        if dir == 'L':
            new_col = col - 1
            if new_col >= 0 and grid[row][new_col] == '.':
                col = new_col
        elif dir == 'R':
            new_col = col + 1
            if new_col < W and grid[row][new_col] == '.':
                col = new_col
        elif dir == 'U':
            new_row = row - 1
            if new_row >= 0 and grid[new_row][col] == '.':
                row = new_row
        elif dir == 'D':
            new_row = row + 1
            if new_row < H and grid[new_row][col] == '.':
                row = new_row
    
    # Convert back to one-based indexing
    final_i = row + 1
    final_j = col + 1
    print(final_i, final_j)

if __name__ == "__main__":
    main()