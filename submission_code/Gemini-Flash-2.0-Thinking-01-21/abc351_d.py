import collections

def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w
        
    def is_magnet(r, c):
        return is_valid(r, c) and grid[r][c] == '#'
        
    def get_adjacent_cells(r, c):
        adj = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc):
                adj.append((nr, nc))
        return adj
        
    move_enabled_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                is_enabled = True
                for nr, nc in get_adjacent_cells(r, c):
                    if is_magnet(nr, nc):
                        is_enabled = False
                        break
                if is_enabled:
                    move_enabled_cells.append((r, c))
                    
    if not move_enabled_cells:
        max_degree_of_freedom = 1
    else:
        max_degree_of_freedom = 0
        for start_r, start_c in move_enabled_cells:
            reachable_cells = set()
            q = collections.deque([(start_r, start_c)])
            reachable_cells.add((start_r, start_c))
            
            while q:
                r, c = q.popleft()
                for nr, nc in get_adjacent_cells(r, c):
                    if (nr, nc) in move_enabled_cells and (nr, nc) not in reachable_cells:
                        reachable_cells.add((nr, nc))
                        q.append((nr, nc))
                        
            max_degree_of_freedom = max(max_degree_of_freedom, len(reachable_cells))
            
    print(max_degree_of_freedom)

if __name__ == '__main__':
    solve()