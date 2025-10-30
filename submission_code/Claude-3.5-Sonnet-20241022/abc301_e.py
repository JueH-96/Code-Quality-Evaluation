from collections import deque
import sys

def solve():
    H, W, T = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    
    # Find start, goal and candies
    start = None
    goal = None
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))

    # BFS state: (row, col, time, candy_state, visited_candies)
    # candy_state is a bit mask of visited candies
    dirs = [(0,1), (0,-1), (1,0), (-1,0)]
    
    def bfs(must_visit):
        visited = set()
        q = deque([(start[0], start[1], 0, 0, 0)])
        visited.add((start[0], start[1], 0))
        
        while q:
            r, c, t, state, cnt = q.popleft()
            
            if t > T:
                continue
                
            if (r,c) == goal and state == must_visit:
                return True
                
            for dr, dc in dirs:
                nr = r + dr
                nc = c + dc
                
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                    new_state = state
                    new_cnt = cnt
                    
                    # If it's a candy position, mark it as visited in state
                    if (nr,nc) in candies:
                        candy_idx = candies.index((nr,nc))
                        if not (state & (1 << candy_idx)):
                            new_cnt += 1
                        new_state |= (1 << candy_idx)
                    
                    if (nr,nc,new_state) not in visited:
                        visited.add((nr,nc,new_state))
                        q.append((nr, nc, t+1, new_state, new_cnt))
        
        return False

    # Try each possible subset of candies
    n = len(candies)
    ans = -1
    
    for mask in range(1 << n):
        if bfs(mask):
            ans = max(ans, bin(mask).count('1'))
            
    print(ans)

solve()