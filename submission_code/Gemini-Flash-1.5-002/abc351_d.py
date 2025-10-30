import collections

def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    q = collections.deque()
    visited = set()
    max_freedom = 0

    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                q.append((r, c))
                visited.add((r,c))
                count = 0
                
                
                q_bfs = collections.deque([(r,c)])
                visited_bfs = {(r,c)}
                
                while q_bfs:
                    curr_r, curr_c = q_bfs.popleft()
                    count +=1
                    
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == '.' and (nr,nc) not in visited_bfs:
                            
                            is_blocked = False
                            for drr,dcc in [(0,1),(0,-1),(1,0),(-1,0)]:
                                nnr,nnc = nr + drr, nc + dcc
                                if 0 <= nnr < h and 0 <= nnc < w and grid[nnr][nnc] == '#':
                                    is_blocked = True
                                    break
                            if not is_blocked:
                                q_bfs.append((nr,nc))
                                visited_bfs.add((nr,nc))

                max_freedom = max(max_freedom, count)

    print(max_freedom)

solve()