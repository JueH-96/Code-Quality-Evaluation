import collections

def solve():
    h, w = map(int, input().split())
    s_grid = [input() for _ in range(h)]
    grid = []
    for i in range(h):
        row = []
        for j in range(w):
            row.append(s_grid[i][j])
        grid.append(row)
    
    target_chars = ['s', 'n', 'u', 'k', 'e']
    
    if grid[0][0] != target_chars[0]:
        print("No")
        return
        
    q = collections.deque([(0, 0, 0)]) # (row, col, char_index)
    visited = set([(0, 0, 0)])
    
    while q:
        r, c, char_index = q.popleft()
        
        if r == h - 1 and c == w - 1:
            print("Yes")
            return
            
        next_char_index = (char_index + 1) % 5
        next_target_char = target_chars[next_char_index]
        
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        for nr, nc in neighbors:
            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr][nc] == next_target_char:
                    if (nr, nc, next_char_index) not in visited:
                        visited.add((nr, nc, next_char_index))
                        q.append((nr, nc, next_char_index))
                        
    print("No")

if __name__ == '__main__':
    solve()