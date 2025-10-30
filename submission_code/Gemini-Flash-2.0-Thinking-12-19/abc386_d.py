import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    given_cells = []
    for _ in range(m):
        x, y, c = sys.stdin.readline().split()
        given_cells.append(((int(x), int(y)), c))
    
    row_black_cols = [[] for _ in range(n + 1)]
    row_white_cols = [[] for _ in range(n + 1)]
    col_black_rows = [[] for _ in range(n + 1)]
    col_white_rows = [[] for _ in range(n + 1)]
    
    for (x, y), color in given_cells:
        if color == 'B':
            row_black_cols[x].append(y)
            col_black_rows[y].append(x)
        else:
            row_white_cols[x].append(y)
            col_white_rows[y].append(x)
            
    row_cutoffs = {}
    col_cutoffs = {}
    
    for r in range(1, n + 1):
        max_black_col = 0
        if row_black_cols[r]:
            max_black_col = max(row_black_cols[r])
        min_white_col = n + 1
        if row_white_cols[r]:
            min_white_col = min(row_white_cols[r])
            
        lower_bound = max_black_col
        upper_bound = min_white_col - 1
        
        if lower_bound > upper_bound:
            if min_white_col <= n:
                print("No")
                return
            if max_black_col > n:
                print("No")
                return
                
        row_cutoffs[r] = max(0, lower_bound)
        
    for c in range(1, n + 1):
        max_black_row = 0
        if col_black_rows[c]:
            max_black_row = max(col_black_rows[c])
        min_white_row = n + 1
        if col_white_rows[c]:
            min_white_row = min(col_white_rows[c])
            
        lower_bound = max_black_row
        upper_bound = min_white_row - 1
        
        if lower_bound > upper_bound:
            if min_white_row <= n:
                print("No")
                return
            if max_black_row > n:
                print("No")
                return
                
        col_cutoffs[c] = max(0, lower_bound)
        
    for r in range(1, n + 1):
        if r not in row_cutoffs:
            row_cutoffs[r] = 0
    for c in range(1, n + 1):
        if c not in col_cutoffs:
            col_cutoffs[c] = 0
            
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            row_color = 'B' if c <= row_cutoffs[r] else 'W'
            col_color = 'B' if r <= col_cutoffs[c] else 'W'
            if row_color != col_color:
                print("No")
                return
                
    for (x, y), color in given_cells:
        expected_color = 'B' if y <= row_cutoffs[x] else 'W'
        if expected_color != color:
            print("No")
            return
            
    print("Yes")

if __name__ == '__main__':
    solve()