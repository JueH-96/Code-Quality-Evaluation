import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    pieces = []
    for _ in range(M):
        r, c = map(int, sys.stdin.readline().split())
        pieces.append((r, c))

    # Sets to store unique attacked rows, columns, major diagonals (sums), and minor diagonals (differences)
    rows_to_exclude = set()
    cols_to_exclude = set()
    sums_to_exclude = set()
    diffs_to_exclude = set()

    for r, c in pieces:
        rows_to_exclude.add(r)
        cols_to_exclude.add(c)
        sums_to_exclude.add(r + c)
        diffs_to_exclude.add(r - c)

    # Principle of Inclusion-Exclusion (PIE) to count total attacked squares
    # total_attacked = Sum(|X|) - Sum(|X intersect Y|) + Sum(|X intersect Y intersect Z|) - |X intersect Y intersect Z intersect W|

    # --- Term 1: Sum of individual set sizes (sum1) ---
    sum1 = 0
    
    # Squares attacked by rows
    sum1 += len(rows_to_exclude) * N
    
    # Squares attacked by columns
    sum1 += len(cols_to_exclude) * N
    
    # Squares attacked by major diagonals (sum = r+c)
    # Number of cells on a diagonal s: N - abs(s - (N + 1))
    for s_val in sums_to_exclude:
        sum1 += (N - abs(s_val - (N + 1)))
        
    # Squares attacked by minor diagonals (diff = r-c)
    # Number of cells on a diagonal d: N - abs(d)
    for d_val in diffs_to_exclude:
        sum1 += (N - abs(d_val))

    # --- Term 2: Sum of 2-way intersections (sum2) ---
    sum2 = 0
    
    # R_U intersect C_U: r in R, c in C
    sum2 += len(rows_to_exclude) * len(cols_to_exclude)

    # R_U intersect S_U: r in R, r+c in S => c = s_val - r_val
    for r_val in rows_to_exclude:
        for s_val in sums_to_exclude:
            c_val = s_val - r_val
            if 1 <= c_val <= N: # Check if (r_val, c_val) is within grid bounds
                sum2 += 1

    # R_U intersect D_U: r in R, r-c in D => c = r_val - d_val
    for r_val in rows_to_exclude:
        for d_val in diffs_to_exclude:
            c_val = r_val - d_val
            if 1 <= c_val <= N:
                sum2 += 1

    # C_U intersect S_U: c in C, r+c in S => r = s_val - c_val
    for c_val in cols_to_exclude:
        for s_val in sums_to_exclude:
            r_val = s_val - c_val
            if 1 <= r_val <= N:
                sum2 += 1

    # C_U intersect D_U: c in C, r-c in D => r = c_val + d_val
    for c_val in cols_to_exclude:
        for d_val in diffs_to_exclude:
            r_val = c_val + d_val
            if 1 <= r_val <= N:
                sum2 += 1

    # S_U intersect D_U: r+c in S, r-c in D => r=(s+d)/2, c=(s-d)/2
    for s_val in sums_to_exclude:
        for d_val in diffs_to_exclude:
            # r and c must be integers, so s_val and d_val must have same parity (s+d must be even)
            if (s_val + d_val) % 2 == 0:
                r_val = (s_val + d_val) // 2
                c_val = (s_val - d_val) // 2
                if 1 <= r_val <= N and 1 <= c_val <= N:
                    sum2 += 1

    # --- Term 3: Sum of 3-way intersections (sum3) ---
    sum3 = 0
    
    # R_U intersect C_U intersect S_U: r in R, c in C, r+c in S
    for r_val in rows_to_exclude:
        for c_val in cols_to_exclude:
            if (r_val + c_val) in sums_to_exclude:
                sum3 += 1
    
    # R_U intersect C_U intersect D_U: r in R, c in C, r-c in D
    for r_val in rows_to_exclude:
        for c_val in cols_to_exclude:
            if (r_val - c_val) in diffs_to_exclude:
                sum3 += 1

    # R_U intersect S_U intersect D_U: r in R, r+c in S, r-c in D
    # Unique (r,c) defined by (s,d), check if r is in R and c is valid
    for s_val in sums_to_exclude:
        for d_val in diffs_to_exclude:
            if (s_val + d_val) % 2 == 0:
                r_val = (s_val + d_val) // 2
                c_val = (s_val - d_val) // 2
                if r_val in rows_to_exclude and 1 <= r_val <= N and 1 <= c_val <= N:
                    sum3 += 1

    # C_U intersect S_U intersect D_U: c in C, r+c in S, r-c in D
    # Unique (r,c) defined by (s,d), check if c is in C and r is valid
    for s_val in sums_to_exclude:
        for d_val in diffs_to_exclude:
            if (s_val - d_val) % 2 == 0:
                c_val = (s_val - d_val) // 2
                r_val = (s_val + d_val) // 2
                if c_val in cols_to_exclude and 1 <= r_val <= N and 1 <= c_val <= N:
                    sum3 += 1

    # --- Term 4: 4-way intersection (sum4) ---
    sum4 = 0
    
    # R_U intersect C_U intersect S_U intersect D_U: r in R, c in C, r+c in S, r-c in D
    for r_val in rows_to_exclude:
        for c_val in cols_to_exclude:
            if (r_val + c_val) in sums_to_exclude and \
               (r_val - c_val) in diffs_to_exclude:
                sum4 += 1

    # Total attacked squares = sum1 - sum2 + sum3 - sum4
    total_attacked_squares = sum1 - sum2 + sum3 - sum4
    
    # Total squares in the N x N grid
    total_squares = N * N
    
    # Safe squares = Total squares - Total attacked squares
    result = total_squares - total_attacked_squares
    
    sys.stdout.write(str(result) + '
')

solve()