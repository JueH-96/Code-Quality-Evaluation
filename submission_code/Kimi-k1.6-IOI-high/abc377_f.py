import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    R_rows = set()
    C_cols = set()
    D_sum = set()
    D_diff = set()
    
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        R_rows.add(a)
        C_cols.add(b)
        D_sum.add(a + b)
        D_diff.add(a - b)
    
    R = len(R_rows)
    C = len(C_cols)
    candidate = (N - R) * (N - C)
    
    R_rows_sorted = sorted(R_rows)
    C_cols_sorted = sorted(C_cols)
    R_rows_set = R_rows
    C_cols_set = C_cols
    
    X = 0
    for s in D_sum:
        lower_j = max(1, s - N)
        upper_j = min(N, s - 1)
        if lower_j > upper_j:
            continue
        total_j = upper_j - lower_j + 1
        
        # Compute A
        A = bisect.bisect_right(C_cols_sorted, upper_j) - bisect.bisect_left(C_cols_sorted, lower_j)
        
        # Compute B
        i_lower = s - upper_j
        i_upper = s - lower_j
        i_lower = max(1, i_lower)
        i_upper = min(N, i_upper)
        if i_lower > i_upper:
            B = 0
        else:
            B = bisect.bisect_right(R_rows_sorted, i_upper) - bisect.bisect_left(R_rows_sorted, i_lower)
        
        # Compute C
        C = 0
        for c in C_cols_sorted:
            if c < lower_j or c > upper_j:
                continue
            i = s - c
            if i in R_rows_set and 1 <= i <= N:
                C += 1
        
        valid_j = total_j - A - B + C
        X += valid_j
    
    Y = 0
    for d in D_diff:
        lower = max(1, 1 - d)
        upper = min(N, N - d)
        if lower > upper:
            continue
        total = upper - lower + 1
        
        # Compute A
        A = bisect.bisect_right(C_cols_sorted, upper) - bisect.bisect_left(C_cols_sorted, lower)
        
        # Compute B
        target_lower = lower + d
        target_upper = upper + d
        target_lower = max(1, target_lower)
        target_upper = min(N, target_upper)
        if target_lower > target_upper:
            B = 0
        else:
            B = bisect.bisect_right(R_rows_sorted, target_upper) - bisect.bisect_left(R_rows_sorted, target_lower)
        
        # Compute C
        C = 0
        for c in C_cols_sorted:
            if c < lower or c > upper:
                continue
            i = c + d
            if i in R_rows_set and 1 <= i <= N:
                C += 1
        
        valid_j = total - A - B + C
        Y += valid_j
    
    Z = 0
    D_sum_list = list(D_sum)
    D_diff_list = list(D_diff)
    for s in D_sum_list:
        for d in D_diff_list:
            if (s + d) % 2 != 0:
                continue
            i = (s + d) // 2
            j = (s - d) // 2
            if 1 <= i <= N and 1 <= j <= N:
                if i not in R_rows_set and j not in C_cols_set:
                    Z += 1
    
    answer = candidate - (X + Y - Z)
    print(answer)

if __name__ == '__main__':
    main()