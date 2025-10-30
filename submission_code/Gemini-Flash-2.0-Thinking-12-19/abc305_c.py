def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    cookie_positions = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#':
                cookie_positions.append((i + 1, j + 1))
    
    if not cookie_positions:
        return
        
    min_row = min(pos[0] for pos in cookie_positions)
    max_row = max(pos[0] for pos in cookie_positions)
    min_col = min(pos[1] for pos in cookie_positions)
    max_col = max(pos[1] for pos in cookie_positions)
    
    eaten_cookie_location = None
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i-1][j-1] == '.':
                eaten_cookie_location = (i, j)
                break
        if eaten_cookie_location:
            break
            
    if eaten_cookie_location:
        print(eaten_cookie_location[0], eaten_cookie_location[1])

if __name__ == '__main__':
    solve()