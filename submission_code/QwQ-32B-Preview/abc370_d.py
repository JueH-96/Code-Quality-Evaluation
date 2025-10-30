import sys
import bisect

def main():
    import sys
    import bisect
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    queries = list(map(int, data[3:]))
    
    rows = [[] for _ in range(H)]
    columns = [[] for _ in range(W)]
    rows_removed = [set() for _ in range(H)]
    columns_removed = [set() for _ in range(W)]
    
    for r in range(H):
        for c in range(1, W+1):
            rows[r].append(c)
    for c in range(W):
        for r in range(1, H+1):
            columns[c].append(r)
    
    total_destroyed = 0
    idx = 0
    for _ in range(Q):
        R_q = queries[idx] - 1  # 0-indexed
        C_q = queries[idx + 1] - 1  # 0-indexed
        idx += 2
        
        if (C_q + 1) in rows_removed[R_q]:
            # Wall exists at (R_q, C_q)
            destroy(rows, columns, rows_removed, columns_removed, R_q, C_q)
            total_destroyed += 1
        else:
            # No wall at (R_q, C_q)
            # Destroy left, right, up, down
            left_j = find_left(rows[R_q], C_q + 1, rows_removed[R_q])
            if left_j is not None:
                destroy(rows, columns, rows_removed, columns_removed, R_q, left_j - 1)
                total_destroyed += 1
            right_j = find_right(rows[R_q], C_q + 1, rows_removed[R_q])
            if right_j is not None:
                destroy(rows, columns, rows_removed, columns_removed, R_q, right_j - 1)
                total_destroyed += 1
            up_i = find_up(columns[C_q], R_q + 1, columns_removed[C_q])
            if up_i is not None:
                destroy(rows, columns, rows_removed, columns_removed, up_i - 1, C_q)
                total_destroyed += 1
            down_i = find_down(columns[C_q], R_q + 1, columns_removed[C_q])
            if down_i is not None:
                destroy(rows, columns, rows_removed, columns_removed, down_i - 1, C_q)
                total_destroyed += 1
    
    remaining_walls = H * W - total_destroyed
    print(remaining_walls)

def destroy(rows, columns, rows_removed, columns_removed, r, c):
    rows_removed[r].add(c + 1)
    columns_removed[c].add(r + 1)

def find_left(row_list, C_q, removed_set):
    idx = bisect.bisect_left(row_list, C_q)
    if idx == 0:
        return None
    idx -= 1
    j = row_list[idx]
    while j in removed_set:
        idx -= 1
        if idx < 0:
            return None
        j = row_list[idx]
    return j

def find_right(row_list, C_q, removed_set):
    idx = bisect.bisect_left(row_list, C_q)
    if idx == len(row_list):
        return None
    j = row_list[idx]
    while j in removed_set:
        idx += 1
        if idx == len(row_list):
            return None
        j = row_list[idx]
    return j

def find_up(column_list, R_q, removed_set):
    idx = bisect.bisect_left(column_list, R_q)
    if idx == 0:
        return None
    idx -= 1
    i = column_list[idx]
    while i in removed_set:
        idx -= 1
        if idx < 0:
            return None
        i = column_list[idx]
    return i

def find_down(column_list, R_q, removed_set):
    idx = bisect.bisect_left(column_list, R_q)
    if idx == len(column_list):
        return None
    i = column_list[idx]
    while i in removed_set:
        idx += 1
        if idx == len(column_list):
            return None
        i = column_list[idx]
    return i

if __name__ == '__main__':
    main()