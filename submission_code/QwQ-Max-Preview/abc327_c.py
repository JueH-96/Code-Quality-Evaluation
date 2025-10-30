import sys

def main():
    grid = []
    for _ in range(9):
        row = list(map(int, sys.stdin.readline().split()))
        grid.append(row)
    
    required = set(range(1, 10))
    
    # Check each row
    for row in grid:
        if set(row) != required:
            print("No")
            return
    
    # Check each column
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        if set(column) != required:
            print("No")
            return
    
    # Check each 3x3 subgrid
    for row_start in [0, 3, 6]:
        for col_start in [0, 3, 6]:
            subgrid = []
            for i in range(row_start, row_start + 3):
                for j in range(col_start, col_start + 3):
                    subgrid.append(grid[i][j])
            if set(subgrid) != required:
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()