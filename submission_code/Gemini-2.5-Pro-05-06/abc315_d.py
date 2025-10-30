import sys
from collections import defaultdict

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid_chars = [sys.stdin.readline().strip() for _ in range(H)]

    is_present = [[True for _ in range(W)] for _ in range(H)]
    
    row_counts = [W] * H
    col_counts = [H] * W

    row_color_freqs = [defaultdict(int) for _ in range(H)]
    col_color_freqs = [defaultdict(int) for _ in range(W)]

    row_live_cs = [set(range(W)) for _ in range(H)] 
    col_live_rs = [set(range(H)) for _ in range(W)] 

    for r_idx in range(H):
        for c_idx in range(W):
            char = grid_chars[r_idx][c_idx]
            row_color_freqs[r_idx][char] += 1
            col_color_freqs[c_idx][char] += 1
            
    total_cookies_remaining = H * W
    
    active_rows_to_check_indices = list(range(H)) 
    active_cols_to_check_indices = list(range(W)) 

    while True:
        marked_coords = set()

        for r_idx in active_rows_to_check_indices:
            if row_counts[r_idx] >= 2 and len(row_color_freqs[r_idx]) == 1:
                for c_idx in row_live_cs[r_idx]: 
                    marked_coords.add((r_idx, c_idx))
        
        for c_idx in active_cols_to_check_indices:
            if col_counts[c_idx] >= 2 and len(col_color_freqs[c_idx]) == 1:
                for r_idx in col_live_rs[c_idx]: 
                    marked_coords.add((r_idx, c_idx))

        if not marked_coords:
            break

        next_iter_active_rows_set = set()
        next_iter_active_cols_set = set()

        for r_coord, c_coord in marked_coords:
            if is_present[r_coord][c_coord]: 
                is_present[r_coord][c_coord] = False 
                total_cookies_remaining -= 1
                
                char_removed = grid_chars[r_coord][c_coord]

                row_counts[r_coord] -= 1
                row_color_freqs[r_coord][char_removed] -= 1
                if row_color_freqs[r_coord][char_removed] == 0:
                    del row_color_freqs[r_coord][char_removed]
                row_live_cs[r_coord].remove(c_coord)
                
                col_counts[c_coord] -= 1
                col_color_freqs[c_coord][char_removed] -= 1
                if col_color_freqs[c_coord][char_removed] == 0:
                    del col_color_freqs[c_coord][char_removed]
                col_live_rs[c_coord].remove(r_coord)

                next_iter_active_rows_set.add(r_coord)
                next_iter_active_cols_set.add(c_coord)
        
        active_rows_to_check_indices = list(next_iter_active_rows_set)
        active_cols_to_check_indices = list(next_iter_active_cols_set)

    sys.stdout.write(str(total_cookies_remaining) + "
")

if __name__ == '__main__':
    main()