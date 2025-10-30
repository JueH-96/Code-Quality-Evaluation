def solve():
    grid_strings = [input() for _ in range(8)]
    existing_pieces_pos = []
    for i in range(8):
        for j in range(8):
            if grid_strings[i][j] == '#':
                existing_pieces_pos.append((i + 1, j + 1))
    
    safe_squares_count = 0
    for i in range(1, 9):
        for j in range(1, 9):
            if grid_strings[i-1][j-1] == '.':
                is_safe = True
                for piece_pos in existing_pieces_pos:
                    piece_row, piece_col = piece_pos
                    if piece_row == i or piece_col == j:
                        is_safe = False
                        break
                if is_safe:
                    safe_squares_count += 1
                    
    print(safe_squares_count)

if __name__ == '__main__':
    solve()