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
        grid.append(line)
    
    # Find min and max row indices that contain '#'
    min_row = H
    max_row = -1
    for row in range(H):
        if '#' in grid[row]:
            if row < min_row:
                min_row = row
            if row > max_row:
                max_row = row
    
    # Find min and max column indices that contain '#'
    min_col = W
    max_col = -1
    for col in range(W):
        for row in range(H):
            if grid[row][col] == '#':
                if col < min_col:
                    min_col = col
                if col > max_col:
                    max_col = col
                break  # No need to check other rows for this column
    
    # Iterate through the subgrid to find the '.' cell
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            if grid[row][col] == '.':
                print(row + 1, col + 1)
                return

if __name__ == "__main__":
    main()