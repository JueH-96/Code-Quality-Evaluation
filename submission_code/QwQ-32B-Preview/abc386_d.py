def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    idx = 2
    
    row_max_c = {}
    col_max_r = {}
    
    black_cells = []
    white_cells = []
    
    for _ in range(M):
        r = int(data[idx]) - 1  # 0-indexed
        c = int(data[idx + 1]) - 1  # 0-indexed
        color = data[idx + 2]
        if color == 'B':
            black_cells.append((r, c))
            if r in row_max_c:
                row_max_c[r] = max(row_max_c[r], c)
            else:
                row_max_c[r] = c
            if c in col_max_r:
                col_max_r[c] = max(col_max_r[c], r)
            else:
                col_max_r[c] = r
        else:  # color == 'W'
            white_cells.append((r, c))
        idx += 3
    
    # For rows without black cells, k_r can be from 0 to N
    # For columns without black cells, l_c can be from 0 to N
    
    # Check white cells
    for r, c in white_cells:
        k_r_fixed = r in row_max_c
        l_c_fixed = c in col_max_r
        
        if k_r_fixed and l_c_fixed:
            # Both k_r and l_c are fixed
            k_r = row_max_c[r]
            l_c = col_max_r[c]
            if c <= k_r and r <= l_c:
                print("No")
                return
        elif k_r_fixed:
            # k_r is fixed, l_c is not fixed
            k_r = row_max_c[r]
            if c <= k_r:
                # Must have r > l_c
                # Since l_c is not fixed, set l_c < r
                if l_c_fixed and l_c >= r:
                    print("No")
                    return
            # Else, l_c can be set to any value < r
        elif l_c_fixed:
            # l_c is fixed, k_r is not fixed
            l_c = col_max_r[c]
            if r <= l_c:
                # Must have c > k_r
                # Since k_r is not fixed, set k_r < c
                if k_r_fixed and k_r >= c:
                    print("No")
                    return
            # Else, k_r can be set to any value < c
        else:
            # Neither k_r nor l_c is fixed
            # Can choose k_r < c or l_c < r
            # Always satisfiable
            pass
    
    print("Yes")

if __name__ == '__main__':
    main()