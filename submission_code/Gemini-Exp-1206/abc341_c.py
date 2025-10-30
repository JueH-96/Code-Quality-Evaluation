def solve():
    h, w, n = map(int, input().split())
    t = input()
    grid = [input() for _ in range(h)]
    
    possible_starts = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] == '.':
                possible_starts.append((r, c))
    
    possible_ends = set()
    
    for start_r, start_c in possible_starts:
        curr_r, curr_c = start_r, start_c
        valid_path = True
        for move in t:
            if move == 'L':
                curr_c -= 1
            elif move == 'R':
                curr_c += 1
            elif move == 'U':
                curr_r -= 1
            elif move == 'D':
                curr_r += 1
            
            if not (0 <= curr_r < h and 0 <= curr_c < w and grid[curr_r][curr_c] == '.'):
                valid_path = False
                break
        
        if valid_path:
            possible_ends.add((curr_r, curr_c))
    
    print(len(possible_ends))

solve()