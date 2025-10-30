def solve():
    grid = []
    for _ in range(9):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Check rows
    for i in range(9):
        row_values = set(grid[i])
        if len(row_values) != 9:
            print("No")
            return
        for val in row_values:
            if not 1 <= val <= 9:
                print("No")
                return
                
    # Check columns
    for j in range(9):
        col_values = set()
        for i in range(9):
            col_values.add(grid[i][j])
        if len(col_values) != 9:
            print("No")
            return
        for val in col_values:
            if not 1 <= val <= 9:
                print("No")
                return
                
    # Check 3x3 blocks
    for block_row_index in range(3):
        for block_col_index in range(3):
            block_values = set()
            start_row = block_row_index * 3
            start_col = block_col_index * 3
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    block_values.add(grid[i][j])
            if len(block_values) != 9:
                print("No")
                return
            for val in block_values:
                if not 1 <= val <= 9:
                    print("No")
                    return
                    
    print("Yes")

if __name__ == '__main__':
    solve()