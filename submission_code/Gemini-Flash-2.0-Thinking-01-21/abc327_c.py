def solve():
    grid = []
    for _ in range(9):
        row_str = input().split()
        row_int = [int(x) for x in row_str]
        grid.append(row_int)
    
    target_set = set(range(1, 10))
    
    # Check rows
    for i in range(9):
        row_set = set(grid[i])
        if row_set != target_set:
            print("No")
            return
            
    # Check columns
    for j in range(9):
        col_set = set()
        for i in range(9):
            col_set.add(grid[i][j])
        if col_set != target_set:
            print("No")
            return
            
    # Check 3x3 sub-grids
    for i_block in range(3):
        for j_block in range(3):
            block_set = set()
            for r in range(3 * i_block, 3 * i_block + 3):
                for c in range(3 * j_block, 3 * j_block + 3):
                    block_set.add(grid[r][c])
            if block_set != target_set:
                print("No")
                return
                
    print("Yes")

if __name__ == '__main__':
    solve()