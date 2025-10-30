import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    grid = []
    for _ in range(N):
        s = input[ptr]
        ptr += 1
        row = [1 if c == 'B' else 0 for c in s]
        grid.append(row)
    
    # Build prefix sum array pre_P
    pre_P = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            pre_P[i][j] = pre_P[i-1][j] + pre_P[i][j-1] - pre_P[i-1][j-1] + grid[i-1][j-1]
    
    total_black = pre_P[N][N]
    
    for _ in range(Q):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        D = int(input[ptr])
        ptr += 1
        
        H = C - A + 1
        W = D - B + 1
        
        full_rows = H // N
        rem_rows = H % N
        full_cols = W // N
        rem_cols = W % N
        
        r_start = A % N
        c_start = B % N
        
        # Component 1: full_rows * full_cols * total_black
        comp1 = full_rows * full_cols * total_black
        
        # Component 2: full_rows * sum_col_slice(c_start, rem_cols)
        comp2 = 0
        if rem_cols != 0:
            c_end = (c_start + rem_cols - 1) % N
            if c_start <= c_end:
                sum_col = pre_P[N][c_end + 1] - pre_P[N][c_start]
            else:
                sum_col = (pre_P[N][N] - pre_P[N][c_start]) + (pre_P[N][c_end + 1] - pre_P[N][0])
            comp2 = full_rows * sum_col
        
        # Component 3: full_cols * sum_row_slice(r_start, rem_rows)
        comp3 = 0
        if rem_rows != 0:
            r_end = (r_start + rem_rows - 1) % N
            if r_start <= r_end:
                sum_row = pre_P[r_end + 1][N] - pre_P[r_start][N]
            else:
                sum_row = (pre_P[N][N] - pre_P[r_start][N]) + (pre_P[r_end + 1][N] - pre_P[0][N])
            comp3 = full_cols * sum_row
        
        # Component 4: sum_rect(r_start, c_start, rem_rows, rem_cols)
        comp4 = 0
        if rem_rows != 0 and rem_cols != 0:
            r_end = (r_start + rem_rows - 1) % N
            c_end = (c_start + rem_cols - 1) % N
            
            if r_start <= r_end:
                if c_start <= c_end:
                    # Case 1: single rectangle
                    comp4 = pre_P[r_end + 1][c_end + 1] - pre_P[r_start][c_end + 1] - pre_P[r_end + 1][c_start] + pre_P[r_start][c_start]
                else:
                    # Case 2: right and left parts
                    sum_right = pre_P[r_end + 1][N] - pre_P[r_start][N] - pre_P[r_end + 1][c_start] + pre_P[r_start][c_start]
                    sum_left = pre_P[r_end + 1][c_end + 1] - pre_P[r_start][c_end + 1] - pre_P[r_end + 1][0] + pre_P[r_start][0]
                    comp4 = sum_right + sum_left
            else:
                if c_start <= c_end:
                    # Case 3: bottom and top parts
                    sum_bottom = pre_P[N][c_end + 1] - pre_P[r_start][c_end + 1] - pre_P[N][c_start] + pre_P[r_start][c_start]
                    sum_top = pre_P[r_end + 1][c_end + 1] - pre_P[0][c_end + 1] - pre_P[r_end + 1][c_start] + pre_P[0][c_start]
                    comp4 = sum_bottom + sum_top
                else:
                    # Case 4: four parts
                    sum_bottom_right = pre_P[N][N] - pre_P[r_start][N] - pre_P[N][c_start] + pre_P[r_start][c_start]
                    sum_bottom_left = pre_P[N][c_end + 1] - pre_P[r_start][c_end + 1] - pre_P[N][0] + pre_P[r_start][0]
                    sum_top_right = pre_P[r_end + 1][N] - pre_P[0][N] - pre_P[r_end + 1][c_start] + pre_P[0][c_start]
                    sum_top_left = pre_P[r_end + 1][c_end + 1] - pre_P[0][c_end + 1] - pre_P[r_end + 1][0] + pre_P[0][0]
                    comp4 = sum_bottom_right + sum_bottom_left + sum_top_right + sum_top_left
        
        total = comp1 + comp2 + comp3 + comp4
        print(total)

if __name__ == '__main__':
    main()