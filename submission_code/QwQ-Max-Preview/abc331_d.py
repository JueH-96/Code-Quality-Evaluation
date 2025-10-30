def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    grid = []
    for _ in range(N):
        line = input[ptr]
        ptr += 1
        grid.append(line)
    
    # Precompute column prefix sums and total B count for each row
    col_prefix = []
    total_B_row = []
    for row in grid:
        prefix = [0] * (N + 1)
        total = 0
        for i in range(N):
            prefix[i+1] = prefix[i] + (1 if row[i] == 'B' else 0)
            total = prefix[i+1]
        col_prefix.append(prefix)
        total_B_row.append(total)
    sum_total = sum(total_B_row)

    output = []
    for _ in range(Q):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        D = int(input[ptr])
        ptr += 1

        # Compute rows
        R = C - A + 1
        s_row = A % N
        initial_rows = (N - s_row) % N
        if initial_rows > R:
            initial_rows = R
            remaining_rows_after_initial = 0
        else:
            remaining_rows_after_initial = R - initial_rows
        
        full_cycles = remaining_rows_after_initial // N
        remaining_remaining = remaining_rows_after_initial % N

        # Calculate sum_initial
        sum_initial = 0
        start_row = s_row
        for i in range(initial_rows):
            row = (start_row + i) % N
            # Compute column sum for this row
            columns = D - B + 1
            if columns <= 0:
                continue
            full_cycles_col = columns // N
            remaining_col = columns % N
            sum_full_col = full_cycles_col * total_B_row[row]
            if remaining_col == 0:
                sum_initial += sum_full_col
                continue
            start_col = B % N
            end_col = (start_col + remaining_col -1) % N
            if start_col <= end_col:
                sum_remaining_col = col_prefix[row][end_col +1] - col_prefix[row][start_col]
            else:
                sum_remaining_col = (col_prefix[row][N] - col_prefix[row][start_col]) + (col_prefix[row][end_col +1] - col_prefix[row][0])
            sum_initial += sum_full_col + sum_remaining_col
        
        # Calculate sum_remaining
        sum_remaining = 0
        for i in range(remaining_remaining):
            row = i
            columns = D - B + 1
            if columns <= 0:
                continue
            full_cycles_col = columns // N
            remaining_col = columns % N
            sum_full_col = full_cycles_col * total_B_row[row]
            if remaining_col == 0:
                sum_remaining += sum_full_col
                continue
            start_col = B % N
            end_col = (start_col + remaining_col -1) % N
            if start_col <= end_col:
                sum_remaining_col = col_prefix[row][end_col +1] - col_prefix[row][start_col]
            else:
                sum_remaining_col = (col_prefix[row][N] - col_prefix[row][start_col]) + (col_prefix[row][end_col +1] - col_prefix[row][0])
            sum_remaining += sum_full_col + sum_remaining_col
        
        sum_full = full_cycles * sum_total
        total = sum_initial + sum_remaining + sum_full
        output.append(total)
    
    print('
'.join(map(str, output)))

if __name__ == '__main__':
    main()