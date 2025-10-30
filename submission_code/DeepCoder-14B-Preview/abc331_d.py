def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    grid = []
    for _ in range(N):
        line = data[idx]
        idx += 1
        row = [1 if c == 'B' else 0 for c in line]
        grid.append(row)
    
    # Precompute prefix sums for rows
    prefix_row = [[0] * (N + 1) for _ in range(N)]
    for r in range(N):
        for c in range(1, N + 1):
            prefix_row[r][c] = prefix_row[r][c-1] + grid[r][c-1]
    
    # Precompute prefix sums for columns
    prefix_col = [[0] * (N + 1) for _ in range(N)]
    for c in range(N):
        for r in range(1, N + 1):
            prefix_col[c][r] = prefix_col[c][r-1] + grid[r-1][c]
    
    base_total = sum(sum(row) for row in grid)
    
    results = []
    for _ in range(Q):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        D = int(data[idx])
        idx += 1
        
        R = C - A + 1
        Cols = D - B + 1
        
        full_v = R // N
        rem_v = R % N
        full_h = Cols // N
        rem_h = Cols % N
        
        sum_main = full_v * full_h * base_total
        
        sum_v = 0
        start_row_v = A % N
        for i in range(rem_v):
            r = (start_row_v + i) % N
            start_col = B % N
            end_col = D % N
            if start_col <= end_col:
                sum_for_row = prefix_row[r][end_col + 1] - prefix_row[r][start_col]
            else:
                sum_for_row = (prefix_row[r][N] - prefix_row[r][start_col]) + prefix_row[r][end_col + 1]
            sum_v += sum_for_row
        
        sum_h = 0
        start_col_h = B % N
        for j in range(rem_h):
            c = (start_col_h + j) % N
            start_row = A % N
            end_row = (A + R - 1) % N
            if start_row <= end_row:
                sum_for_col = prefix_col[c][end_row + 1] - prefix_col[c][start_row]
            else:
                sum_for_col = (prefix_col[c][N] - prefix_col[c][start_row]) + prefix_col[c][end_row + 1]
            sum_h += sum_for_col
        
        sum_small = 0
        start_row_v = A % N
        start_col_h = B % N
        for i in range(rem_v):
            r = (start_row_v + i) % N
            for j in range(rem_h):
                c = (start_col_h + j) % N
                sum_small += grid[r][c]
        
        total = sum_main + sum_v + sum_h - sum_small
        results.append(total)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()