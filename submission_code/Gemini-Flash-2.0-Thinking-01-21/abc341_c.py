def solve():
    h, w, n = map(int, input().split())
    t = input()
    s = [input() for _ in range(h)]
    
    possible_final_positions = set()
    
    for start_row in range(1, h + 1):
        for start_col in range(1, w + 1):
            if s[start_row-1][start_col-1] == '.':
                current_row, current_col = start_row, start_col
                is_valid_path = True
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
                        
                    if 1 <= next_row <= h and 1 <= next_col <= w and s[next_row-1][next_col-1] == '.':
                        current_row, current_col = next_row, next_col
                    else:
                        is_valid_path = False
                        break
                        
                if is_valid_path:
                    possible_final_positions.add((current_row, current_col))
                    
    print(len(possible_final_positions))

if __name__ == '__main__':
    solve()