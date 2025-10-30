def solve():
    h, w, n = map(int, input().split())
    initial_rows = []
    start_cols = []
    lengths = []
    for _ in range(n):
        r, c, l = map(int, input().split())
        initial_rows.append(r)
        start_cols.append(c)
        lengths.append(l)
    
    current_rows = list(initial_rows)
    
    while True:
        moved_in_step = False
        for i in range(n):
            if current_rows[i] < h:
                can_move_down = True
                for j in range(lengths[i]):
                    cell_row = current_rows[i] + 1
                    cell_col = start_cols[i] + j
                    if cell_row > h:
                        can_move_down = False
                        break
                    is_occupied_below = False
                    for k in range(n):
                        if k == i:
                            continue
                        if current_rows[k] == cell_row and start_cols[k] <= cell_col < start_cols[k] + lengths[k]:
                            is_occupied_below = True
                            break
                    if is_occupied_below:
                        can_move_down = False
                        break
                if can_move_down:
                    current_rows[i] += 1
                    moved_in_step = True
                    
        if not moved_in_step:
            break
            
    for final_row in current_rows:
        print(final_row)

if __name__ == '__main__':
    solve()