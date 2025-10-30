def solve():
    n, m = map(int, input().split())
    occupied_squares = set()
    occupied_list = []
    for _ in range(m):
        u, v = map(int, input().split())
        occupied_squares.add((u, v))
        occupied_list.append((u, v))
    
    unsafe_rows = set()
    unsafe_cols = set()
    unsafe_sums = set()
    unsafe_diffs = set()
    
    for r, c in occupied_list:
        unsafe_rows.add(r)
        unsafe_cols.add(c)
        unsafe_sums.add(r + c)
        unsafe_diffs.add(r - c)
        
    unsafe_squares_count = 0
    
    for i_mask in range(1, 16):
        current_squares = []
        
        def is_unsafe(r, c, mask):
            if (mask & 1) and (r not in unsafe_rows):
                return False
            if (mask & 2) and (c not in unsafe_cols):
                return False
            if (mask & 4) and ((r + c) not in unsafe_sums):
                return False
            if (mask & 8) and ((r - c) not in unsafe_diffs):
                return False
            return True
            
        count_intersection_u_i = 0
        count_intersection_u_i_occupied = 0
        
        for r in range(1, n + 1):
            for c in range(1, n + 1):
                conditions_met = True
                if (i_mask & 1) and (r not in unsafe_rows):
                    conditions_met = False
                if conditions_met and (i_mask & 2) and (c not in unsafe_cols):
                    conditions_met = False
                if conditions_met and (i_mask & 4) and ((r + c) not in unsafe_sums):
                    conditions_met = False
                if conditions_met and (i_mask & 8) and ((r - c) not in unsafe_diffs):
                    conditions_met = False
                    
                if conditions_met:
                    count_intersection_u_i += 1
                    if (r, c) in occupied_squares:
                        count_intersection_u_i_occupied += 1
                        
        n_i = count_intersection_u_i - count_intersection_u_i_occupied
        
        if bin(i_mask).count('1') % 2 == 1:
            unsafe_squares_count += n_i
        else:
            unsafe_squares_count -= n_i
            
    total_empty_squares = n * n - m
    safe_empty_squares = total_empty_squares - unsafe_squares_count
    print(safe_empty_squares)

if __name__ == '__main__':
    solve()