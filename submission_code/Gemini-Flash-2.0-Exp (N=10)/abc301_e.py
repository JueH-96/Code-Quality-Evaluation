from collections import deque

def solve():
    h, w, t = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    start_r, start_c = -1, -1
    goal_r, goal_c = -1, -1
    candies = []
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_r, start_c = r, c
            elif grid[r][c] == 'G':
                goal_r, goal_c = r, c
            elif grid[r][c] == 'o':
                candies.append((r, c))
    
    num_candies = len(candies)
    
    def bfs(start_r, start_c, end_r, end_c, max_moves):
        q = deque([(start_r, start_c, 0, 0)])  # (row, col, moves, candy_mask)
        visited = set()
        max_candies = -1
        
        while q:
            r, c, moves, candy_mask = q.popleft()
            
            if (r, c) == (end_r, end_c):
                max_candies = max(max_candies, bin(candy_mask).count('1'))
                continue
            
            if moves >= max_moves:
                continue
            
            if (r, c, candy_mask) in visited:
                continue
            visited.add((r, c, candy_mask))
            
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#':
                    new_candy_mask = candy_mask
                    for i, (cr, cc) in enumerate(candies):
                        if nr == cr and nc == cc:
                            new_candy_mask |= (1 << i)
                            break
                    q.append((nr, nc, moves + 1, new_candy_mask))
        return max_candies
    
    result = bfs(start_r, start_c, goal_r, goal_c, t)
    print(result)

solve()