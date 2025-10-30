import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print("No")
        return
    
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    N = int(data[1+H])
    medicines = []
    for i in range(1+H+1, 1+H+1+N):
        parts = data[i].split()
        if len(parts) < 3:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        e = int(parts[2])
        medicines.append((r, c, e))
    
    start = None
    goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
                
    if start is None or goal is None:
        print("No")
        return
        
    if start == goal:
        print("Yes")
        return
        
    nodes = [start]
    for (r, c, e) in medicines:
        nodes.append((r, c))
        
    L = len(nodes)
    INF = 10**9
    dist_node = [[INF] * L for _ in range(L)]
    dist_to_goal = [INF] * L
    
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for idx in range(L):
        r0, c0 = nodes[idx]
        dist_grid = [[-1] * W for _ in range(H)]
        q = deque()
        q.append((r0, c0))
        dist_grid[r0][c0] = 0
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] != '#' and dist_grid[nr][nc] == -1:
                        dist_grid[nr][nc] = dist_grid[r][c] + 1
                        q.append((nr, nc))
                        
        for j in range(L):
            rj, cj = nodes[j]
            d_val = dist_grid[rj][cj]
            if d_val == -1:
                dist_node[idx][j] = INF
            else:
                dist_node[idx][j] = d_val
                
        d_goal = dist_grid[goal[0]][goal[1]]
        if d_goal == -1:
            dist_to_goal[idx] = INF
        else:
            dist_to_goal[idx] = d_goal
            
    e_list = [0] * L
    for i in range(1, L):
        e_list[i] = medicines[i-1][2]
        
    reachable = [False] * L
    for i in range(1, L):
        if nodes[0] == nodes[i]:
            reachable[i] = True
            
    D = [0] * L
    q = deque()
    for i in range(1, L):
        if reachable[i]:
            D[i] = e_list[i]
            q.append(i)
            
    while q:
        j = q.popleft()
        for k in range(1, L):
            if k == j:
                continue
            if not reachable[k]:
                if D[j] >= dist_node[j][k]:
                    reachable[k] = True
                    D[k] = e_list[k]
                    q.append(k)
                    
    for i in range(1, L):
        if reachable[i] and D[i] >= dist_to_goal[i]:
            print("Yes")
            return
            
    print("No")

if __name__ == "__main__":
    main()