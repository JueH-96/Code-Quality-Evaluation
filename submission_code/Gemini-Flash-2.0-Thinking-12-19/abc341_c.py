def solve():
    h, w, n = map(int, input().split())
    t = input()
    grid_str = [input() for _ in range(h)]
    is_land = [[(grid_str[i][j] == '.') for j in range(w)] for i in range(h)]
    possible_final_positions = set()
    
    for start_row in range(h):
        for start_col in range(w):
            if is_land[start_row][start_col]:
                current_row, current_col = start_row, start_col
                is_valid_path = True
                path_cells = [(current_row, current_col)]
                for move in t:
                    next_row, next_col = current_row, current_col
                    if move == 'L':
                        next_col -= 1
                    elif move == 'R':
                        next_col += 1
                    elif move == 'U':
                        next_row -= 1
                    elif move == 'D':
                        next_row += 1
                    
                    if 0 <= next_row < h and 0 <= next_col < w and is_land[next_row][next_col]:
                        current_row, current_col = next_row, next_col
                        path_cells.append((current_row, current_col))
                    else:
                        is_valid_path = False
                        break
                        
                if is_valid_path:
                    possible_final_positions.add((current_row, current_col))
                    
    print(len(possible_final_positions))

if __name__ == '__main__':
    solve()