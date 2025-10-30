def solve():
    H, W, N = map(int, input().split())
    T = input()
    grid = [input() for _ in range(H)]

    possible_end_positions = set()

    for start_row in range(H):
        for start_col in range(W):
            
            valid_path = True
            current_row = start_row
            current_col = start_col

            if grid[current_row][current_col] == '#':
                continue
            
            for move in T:
                if move == 'L':
                    current_col -= 1
                elif move == 'R':
                    current_col += 1
                elif move == 'U':
                    current_row -= 1
                elif move == 'D':
                    current_row += 1
                
                if not (0 <= current_row < H and 0 <= current_col < W) or grid[current_row][current_col] == '#':
                    valid_path = False
                    break
            
            if valid_path:
                possible_end_positions.add((current_row, current_col))
    
    print(len(possible_end_positions))

solve()