def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N_Q = data[0].split()
    N = int(N_Q[0])
    Q = int(N_Q[1])
    
    P = [data[i+1] for i in range(N)]
    
    # Precompute count_B: total number of 'B's in P
    count_B = 0
    for row in P:
        count_B += row.count('B')
    
    # Precompute row_prefix[i]: sum of 'B's in rows 0 to i-1
    row_prefix = [0] * (N + 1)
    for i in range(N):
        row_prefix[i+1] = row_prefix[i] + P[i].count('B')
    
    # Precompute col_prefix[j]: sum of 'B's in columns 0 to j-1
    col_prefix = [0] * (N + 1)
    for j in range(N):
        col_sum = 0
        for i in range(N):
            if P[i][j] == 'B':
                col_sum += 1
        col_prefix[j+1] = col_prefix[j] + col_sum
    
    # Precompute 2D prefix sum for P
    prefix = [[0]*(N+1) for _ in range(N+1)]
    for i in range(N):
        row_sum = 0
        for j in range(N):
            row_sum += 1 if P[i][j] == 'B' else 0
            prefix[i+1][j+1] = prefix[i][j+1] + prefix[i+1][j] - prefix[i][j] + row_sum - row_prefix[i]
    
    # Function to calculate sum of row_blacks from a to a+length-1, with wrap-around
    def sum_rows(a, length):
        if a + length <= N:
            return row_prefix[a + length] - row_prefix[a]
        else:
            return (row_prefix[N] - row_prefix[a]) + row_prefix[a + length - N]
    
    # Function to calculate sum of col_blacks from a to a+length-1, with wrap-around
    def sum_cols(a, length):
        if a + length <= N:
            return col_prefix[a + length] - col_prefix[a]
        else:
            return (col_prefix[N] - col_prefix[a]) + col_prefix[a + length - N]
    
    # Function to calculate number of 'B's in sub-rectangle from (x1,y1) to (x2,y2)
    def corner_black(x1, y1, rem_h, rem_v):
        x2 = x1 + rem_h - 1
        y2 = y1 + rem_v - 1
        if x2 >= N or y2 >= N:
            # Handle wrap-around
            # Calculate the number of 'B's in two rectangles
            # First from (x1, y1) to (N-1, y2) if x2 >= N
            # Second from (0, y1) to (x2 - N, y2) if x2 - N >= 0
            pass
        return prefix[x2+1][y2+1] - prefix[x2+1][y1] - prefix[x1][y2+1] + prefix[x1][y1]
    
    # Process each query
    for q in range(Q):
        A, B, C, D = map(int, data[N+1+q].split())
        full_h = (C - A + 1) // N
        full_v = (D - B + 1) // N
        full_blocks = full_h * full_v
        black_from_full_blocks = full_blocks * count_B
        
        rem_h = (C - A + 1) % N
        rem_v = (D - B + 1) % N
        offset_i = A % N
        offset_j = B % N
        
        if rem_h > 0:
            horizontal_black = full_v * sum_rows(offset_i, rem_h)
        else:
            horizontal_black = 0
        
        if rem_v > 0:
            vertical_black = full_h * sum_cols(offset_j, rem_v)
        else:
            vertical_black = 0
        
        if rem_h > 0 and rem_v > 0:
            # Handle corner_black with wrap-around
            corner_b = corner_black(offset_i, offset_j, rem_h, rem_v)
        else:
            corner_b = 0
        
        total_black = black_from_full_blocks + horizontal_black + vertical_black + corner_b
        print(total_black)

if __name__ == '__main__':
    main()