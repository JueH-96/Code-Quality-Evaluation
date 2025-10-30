# YOUR CODE HERE
def solve():
    h, w, n = map(int, input().split())
    t = input()
    grid = [input() for _ in range(h)]
    
    possible_starts = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                possible_starts.append((r, c))
    
    count = 0
    for start_r, start_c in possible_starts:
        r, c = start_r, start_c
        valid = True
        path = [(r, c)]
        
        for move in t:
            if move == 'L':
                c -= 1
            elif move == 'R':
                c += 1
            elif move == 'U':
                r -= 1
            elif move == 'D':
                r += 1
            
            if not (0 <= r < h and 0 <= c < w and grid[r][c] == '.'):
                valid = False
                break
            path.append((r,c))
        
        if valid:
            count += 1

    
    possible_ends = set()
    for start_r, start_c in possible_starts:
        r, c = start_r, start_c
        valid = True
        
        for move in t:
            if move == 'L':
                c -= 1
            elif move == 'R':
                c += 1
            elif move == 'U':
                r -= 1
            elif move == 'D':
                r += 1
            
            if not (0 <= r < h and 0 <= c < w and grid[r][c] == '.'):
                valid = False
                break
        
        if valid:
            possible_ends.add((r,c))
            
    print(len(possible_ends))

solve()