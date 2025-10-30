def main():
    grid = []
    for _ in range(9):
        line = list(map(int, input().split()))
        grid.append(line)
    
    valid_set = set(range(1, 10))
    
    for i in range(9):
        row_set = set(grid[i])
        if row_set != valid_set:
            print("No")
            return
    
    for j in range(9):
        col_list = [grid[i][j] for i in range(9)]
        col_set = set(col_list)
        if col_set != valid_set:
            print("No")
            return
    
    starts = [0, 3, 6]
    for i in starts:
        for j in starts:
            block = []
            for x in range(3):
                for y in range(3):
                    block.append(grid[i+x][j+y])
            block_set = set(block)
            if block_set != valid_set:
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()