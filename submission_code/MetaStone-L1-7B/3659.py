MOD = 10**9 + 7

def countPathsWithXOR(grid, k):
    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])
    if n == 0:
        return 0
    
    prev_row = [dict() for _ in range(n)]
    # Initialize the first row
    for x in grid[0]:
        new_dict = {x: 1}
        prev_row[0][x] = new_dict[x]
    
    for i in range(1, m):
        curr_row = [dict() for _ in range(n)]
        # Process first column
        top_dict = prev_row[0]
        current_value = grid[i][0]
        for x in top_dict:
            new_x = x ^ current_value
            curr_row[0][new_x] = (curr_row[0].get(new_x, 0) + top_dict[x]) % MOD
        # Process other columns
        for j in range(1, n):
            top_dict = prev_row[j]
            left_dict = curr_row[j-1]
            current_value = grid[i][j]
            # From top
            for x in top_dict:
                new_x = x ^ current_value
                curr_row[j][new_x] = (curr_row[j].get(new_x, 0) + top_dict[x]) % MOD
            # From left
            for y in left_dict:
                new_y = y ^ current_value
                curr_row[j][new_y] = (curr_row[j].get(new_y, 0) + left_dict[y]) % MOD
        prev_row = curr_row
    
    if not prev_row:
        return 0
    result = prev_row[n-1].get(k, 0)
    return result % MOD