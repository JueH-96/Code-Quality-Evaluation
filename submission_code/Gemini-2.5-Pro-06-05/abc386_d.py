import sys
from collections import deque

def solve():
    """
    Solves the Grid Coloring problem by iterative constraint propagation.
    """
    sys.setrecursionlimit(4 * 10**5 + 10)

    try:
        input_lines = sys.stdin.read().splitlines()
        if not input_lines:
            # Handle empty input case
            print("Yes")
            return
        
        N_str, M_str = input_lines[0].split()
        N, M = int(N_str), int(M_str)
        
        if M == 0:
            print("Yes")
            return

        constraints = [line.split() for line in input_lines[1:]]
    except (IOError, ValueError):
        # Handle potential empty input or format errors gracefully
        print("Yes") # Or No, depending on problem spec for malformed input
        return


    row_indices = set()
    col_indices = set()
    for x_str, y_str, _ in constraints:
        row_indices.add(int(x_str))
        col_indices.add(int(y_str))

    low_R, up_R = {r: 0 for r in row_indices}, {r: N for r in row_indices}
    low_C, up_C = {c: 0 for c in col_indices}, {c: N for c in col_indices}

    for x_str, y_str, c_char in constraints:
        r, c = int(x_str), int(y_str)
        if c_char == 'B':
            low_R[r] = max(low_R[r], c)
            low_C[c] = max(low_C[c], r)
        else:
            up_R[r] = min(up_R[r], c - 1)
            up_C[c] = min(up_C[c], r - 1)
            
    sorted_rows = sorted(list(row_indices))
    sorted_cols = sorted(list(col_indices))
    
    row_map = {val: i for i, val in enumerate(sorted_rows)}
    col_map = {val: i for i, val in enumerate(sorted_cols)}
    
    # Propagate non-increasing property once initially
    for i in range(len(sorted_rows) - 2, -1, -1):
        r1, r2 = sorted_rows[i], sorted_rows[i+1]
        low_R[r1] = max(low_R[r1], low_R[r2])
    for i in range(len(sorted_rows) - 1):
        r1, r2 = sorted_rows[i], sorted_rows[i+1]
        up_R[r2] = min(up_R[r2], up_R[r1])

    for i in range(len(sorted_cols) - 2, -1, -1):
        c1, c2 = sorted_cols[i], sorted_cols[i+1]
        low_C[c1] = max(low_C[c1], low_C[c2])
    for i in range(len(sorted_cols) - 1):
        c1, c2 = sorted_cols[i], sorted_cols[i+1]
        up_C[c2] = min(up_C[c2], up_C[c1])

    for r in row_indices:
        if low_R[r] > up_R[r]:
            print("No")
            return
    for c in col_indices:
        if low_C[c] > up_C[c]:
            print("No")
            return

    q_R = deque(row_indices)
    q_C = deque(col_indices)
    
    in_q_R = {r: True for r in row_indices}
    in_q_C = {c: True for c in col_indices}
    
    while q_R or q_C:
        while q_R:
            r = q_R.popleft()
            in_q_R[r] = False
            
            idx = row_map[r]
            
            # Propagate to previous row
            if idx > 0:
                prev_r = sorted_rows[idx-1]
                if low_R[prev_r] < low_R[r]:
                    low_R[prev_r] = low_R[r]
                    if not in_q_R.get(prev_r): q_R.append(prev_r); in_q_R[prev_r] = True
                if up_R[r] > up_R[prev_r]:
                    up_R[r] = up_R[prev_r]
                    if not in_q_R.get(r): q_R.append(r); in_q_R[r] = True
            
            # Propagate to next row
            if idx < len(sorted_rows) - 1:
                next_r = sorted_rows[idx+1]
                if low_R[r] > low_R[next_r]:
                    low_R[next_r] = low_R[r]
                    if not in_q_R.get(next_r): q_R.append(next_r); in_q_R[next_r] = True
                if up_R[next_r] > up_R[r]:
                    up_R[next_r] = up_R[r]
                    if not in_q_R.get(next_r): q_R.append(next_r); in_q_R[next_r] = True


            if low_R[r] > up_R[r]: print("No"); return

            # Duality R -> C
            k_col = low_R[r]
            if k_col in col_indices:
                if low_C[k_col] < r:
                    low_C[k_col] = r
                    if not in_q_C.get(k_col): q_C.append(k_col); in_q_C[k_col] = True

            k_col = up_R[r] + 1
            if k_col in col_indices:
                if up_C[k_col] > r - 1:
                    up_C[k_col] = r - 1
                    if not in_q_C.get(k_col): q_C.append(k_col); in_q_C[k_col] = True

        while q_C:
            c = q_C.popleft()
            in_q_C[c] = False
            
            idx = col_map[c]

            # Propagate to previous column
            if idx > 0:
                prev_c = sorted_cols[idx-1]
                if low_C[prev_c] < low_C[c]:
                    low_C[prev_c] = low_C[c]
                    if not in_q_C.get(prev_c): q_C.append(prev_c); in_q_C[prev_c] = True
                if up_C[c] > up_C[prev_c]:
                    up_C[c] = up_C[prev_c]
                    if not in_q_C.get(c): q_C.append(c); in_q_C[c] = True

            # Propagate to next column
            if idx < len(sorted_cols) - 1:
                next_c = sorted_cols[idx+1]
                if low_C[c] > low_C[next_c]:
                     low_C[next_c] = low_C[c]
                     if not in_q_C.get(next_c): q_C.append(next_c); in_q_C[next_c] = True
                if up_C[next_c] > up_C[c]:
                    up_C[next_c] = up_C[c]
                    if not in_q_C.get(next_c): q_C.append(next_c); in_q_C[next_c] = True

            if low_C[c] > up_C[c]: print("No"); return

            # Duality C -> R
            k_row = low_C[c]
            if k_row in row_indices:
                if low_R[k_row] < c:
                    low_R[k_row] = c
                    if not in_q_R.get(k_row): q_R.append(k_row); in_q_R[k_row] = True

            k_row = up_C[c] + 1
            if k_row in row_indices:
                if up_R[k_row] > c - 1:
                    up_R[k_row] = c - 1
                    if not in_q_R.get(k_row): q_R.append(k_row); in_q_R[k_row] = True

    print("Yes")

solve()