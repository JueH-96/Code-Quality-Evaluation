def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # Read grid dimensions
    H, W = map(int, data[0].split())
    grid = data[1:]
    
    # Determine the minimal bounding box (row and column indices) that covers all already painted black cells.
    min_row, max_row = H, -1
    min_col, max_col = W, -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < min_row:
                    min_row = i
                if i > max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
    
    # Check if any cell inside the bounding box is forced to be white ('.')
    # If yes, then it is impossible to form a filled rectangle.
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                print("No")
                return
    
    # Otherwise, we can color all '?' as needed so that the entire bounding box becomes black 
    # and the remaining area white.
    print("Yes")

if __name__ == '__main__':
    main()