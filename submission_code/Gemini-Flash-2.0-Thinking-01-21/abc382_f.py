def solve():
    h, w, n = map(int, input().split())
    bars_init_data = []
    for _ in range(n):
        r, c, l = map(int, input().split())
        bars_init_data.append({'r': r - 1, 'c': c - 1, 'l': l})
    
    bar_rows = [data['r'] for data in bars_init_data]
    bar_cols = [data['c'] for data in bars_init_data]
    bar_lengths = [data['l'] for data in bars_init_data]
    
    occupied_cols_in_row = [set() for _ in range(h)]
    for i in range(n):
        for j in range(bar_cols[i], bar_cols[i] + bar_lengths[i]):
            occupied_cols_in_row[bar_rows[i]].add(j)
            
    moved_in_step = True
    while moved_in_step:
        moved_in_step = False
        for i in range(n):
            current_row = bar_rows[i]
            if current_row < h - 1:
                can_move_down = True
                for j in range(bar_cols[i], bar_cols[i] + bar_lengths[i]):
                    if j in occupied_cols_in_row[current_row + 1]:
                        can_move_down = False
                        break
                if can_move_down:
                    for j in range(bar_cols[i], bar_cols[i] + bar_lengths[i]):
                        occupied_cols_in_row[current_row].remove(j)
                        occupied_cols_in_row[current_row + 1].add(j)
                    bar_rows[i] += 1
                    moved_in_step = True
                    
    result_rows = [row + 1 for row in bar_rows]
    for row in result_rows:
        print(row)

if __name__ == '__main__':
    solve()