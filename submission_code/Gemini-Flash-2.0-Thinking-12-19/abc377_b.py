def solve():
    grid_strings = [input() for _ in range(8)]
    row_has_piece = [False] * 8
    col_has_piece = [False] * 8
    
    for i in range(8):
        if '#' in grid_strings[i]:
            row_has_piece[i] = True
            
    for j in range(8):
        for i in range(8):
            if grid_strings[i][j] == '#':
                col_has_piece[j] = True
                break
                
    count = 0
    for i in range(8):
        for j in range(8):
            if grid_strings[i][j] == '.':
                if not row_has_piece[i] and not col_has_piece[j]:
                    count += 1
                    
    print(count)

if __name__ == '__main__':
    solve()