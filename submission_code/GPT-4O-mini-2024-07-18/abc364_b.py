# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    S_i, S_j = map(int, data[1].split())
    grid = [data[i + 2] for i in range(H)]
    X = data[H + 2]
    
    # Convert to zero-based index
    current_row = S_i - 1
    current_col = S_j - 1
    
    for move in X:
        if move == 'L':
            if current_col > 0 and grid[current_row][current_col - 1] == '.':
                current_col -= 1
        elif move == 'R':
            if current_col < W - 1 and grid[current_row][current_col + 1] == '.':
                current_col += 1
        elif move == 'U':
            if current_row > 0 and grid[current_row - 1][current_col] == '.':
                current_row -= 1
        elif move == 'D':
            if current_row < H - 1 and grid[current_row + 1][current_col] == '.':
                current_row += 1
    
    # Convert back to one-based index for output
    print(current_row + 1, current_col + 1)

if __name__ == "__main__":
    main()