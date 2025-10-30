def main():
    grid = [list(map(int, input().split())) for _ in range(9)]
    standard_set = set(range(1, 10))
    
    # Check rows
    for row in grid:
        if set(row) != standard_set:
            print("No")
            return
    
    # Check columns
    for col in zip(*grid):
        if set(col) != standard_set:
            print("No")
            return
    
    # Check 3x3 subgrids
    for i in range(0,9,3):
        for j in range(0,9,3):
            subgrid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if set(subgrid) != standard_set:
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()