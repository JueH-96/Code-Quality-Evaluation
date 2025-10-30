import sys

def main():
    grid = [list(map(int, input().split())) for _ in range(9)]
    
    # Check each row
    for row in grid:
        if len(set(row)) != 9:
            print("No")
            return
    
    # Check each column
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        if len(set(column)) != 9:
            print("No")
            return
    
    # Check each 3x3 subgrid
    for i in (0, 3, 6):
        for j in (0, 3, 6):
            block = []
            for x in range(i, i+3):
                for y in range(j, j+3):
                    block.append(grid[x][y])
            if len(set(block)) != 9:
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()