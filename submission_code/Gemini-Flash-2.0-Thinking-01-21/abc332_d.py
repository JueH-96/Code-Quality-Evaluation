import collections

def get_rows(grid):
    return [tuple(row) for row in grid]

def get_cols(grid):
    cols = []
    for j in range(len(grid[0])):
        col = []
        for i in range(len(grid)):
            col.append(grid[i][j])
        cols.append(tuple(col))
    return cols

def get_row_multisets(grid):
    multisets = []
    for row in grid:
        multisets.append(tuple(sorted(row)))
    return sorted(multisets)

def get_col_multisets(grid):
    multisets = []
    cols = get_cols(grid)
    for col in cols:
        multisets.append(tuple(sorted(col)))
    return sorted(multisets)

def get_inversions(p):
    count = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                count += 1
    return count

def apply_row_permutation(grid, p):
    n_rows = len(grid)
    new_grid = [None] * n_rows
    for i in range(n_rows):
        new_grid[i] = grid[p[i]-1]
    return new_grid

def apply_col_permutation(grid, p):
    n_rows = len(grid)
    n_cols = len(grid[0])
    new_grid = [([0] * n_cols) for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            new_grid[i][j] = grid[i][p[j]-1]
    return new_grid

def are_grids_equal(grid1, grid2):
    if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
        return False
    for i in range(len(grid1)):
        if tuple(grid1[i]) != tuple(grid2[i]):
            return False
    return True

def solve():
    h, w = map(int, input().split())
    a_grid = []
    for _ in range(h):
        a_grid.append(list(map(int, input().split())))
    b_grid = []
    for _ in range(h):
        b_grid.append(list(map(int, input().split())))
        
    if get_row_multisets(a_grid) != get_row_multisets(b_grid):
        print("-1")
        return
    if get_col_multisets(a_grid) != get_col_multisets(b_grid):
        print("-1")
        return
        
    import itertools
    row_permutations = list(itertools.permutations(range(1, h + 1)))
    col_permutations = list(itertools.permutations(range(1, w + 1)))
    
    min_ops = float('inf')
    found_solution = False
    
    for row_perm_tuple in row_permutations:
        row_perm = list(row_perm_tuple)
        row_inv = get_inversions(row_perm)
        grid_after_rows = apply_row_permutation(a_grid, row_perm)
        for col_perm_tuple in col_permutations:
            col_perm = list(col_perm_tuple)
            col_inv = get_inversions(col_perm)
            final_grid = apply_col_permutation(grid_after_rows, col_perm)
            if are_grids_equal(final_grid, b_grid):
                min_ops = min(min_ops, row_inv + col_inv)
                found_solution = True
                
    if found_solution:
        print(min_ops)
    else:
        print("-1")

if __name__ == '__main__':
    solve()