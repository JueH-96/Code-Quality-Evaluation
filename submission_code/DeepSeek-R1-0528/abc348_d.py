import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    s_r, s_c = -1, -1
    t_r, t_c = -1, -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                s_r, s_c = i, j
            if grid[i][j] == 'T':
                t_r, t_c = i, j
                
    if s_r == -1 or t_r == -1:
        print("No")
        return
        
    N = int(data[1+H])
    medicines = []
    med_dict = {}
    start_index = 1+H+1
    for i in range(N):
        parts = data[start_index+i].split()
        if len(parts) < 3:
            continue
        r = int(parts[0]); c = int(parts[1]); e = int(parts[2])
        r0 = r-1; c0 = c-1
        medicines.append((r0, c0, e))
        med_dict[(r0, c0)] = e
        
    nodes = []
    nodes.append((s_r, s_c))
    nodes.append((t_r, t_c))
    for (r0, c0, e) in medicines:
        if (r0, c0) == (s_r, s_c) or (r0, c0) == (t_r, t_c):
            continue
        else:
            nodes.append((r0, c0))
            
    n_nodes = len(nodes)
    med_value = []
    for (r, c) in nodes:
        if (r, c) in med_dict:
            med_value.append(med_dict[(r, c)])
        else:
            med_value.append(-1)
            
    if s_r == t_r and s_c == t_c:
        if med_value[0] != -1:
            if med_value[0] >= 0:
                print("Yes")
                return
        else:
            if 0 >= 0:
                print("Yes")
                return
        print("No")
        return

    D = [[-1] * n_nodes for _ in range(n_nodes)]
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for idx in range(n_nodes):
        r0, c0 = nodes[idx]
        if grid[r0][c0] == '#':
            continue
            
        dist_grid = [[-1] * W for _ in range(H)]
        q = deque()
        dist_grid[r0][c0] = 0
        q.append((r0, c0))
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#' or dist_grid[nr][nc] != -1:
                        continue
                    dist_grid[nr][nc] = dist_grid[r][c] + 1
                    q.append((nr, nc))
                    
        for j in range(n_nodes):
            r1, c1 = nodes[j]
            D[idx][j] = dist_grid[r1][c1]
            
    energy = [-10**9] * n_nodes
    if med_value[0] != -1:
        energy[0] = med_value[0]
    else:
        energy[0] = 0
        
    n = n_nodes
    updated = True
    for it in range(n):
        updated = False
        for u in range(n):
            if energy[u] < 0:
                continue
            for v in range(n):
                if u == v:
                    continue
                if D[u][v] == -1:
                    continue
                if energy[u] < D[u][v]:
                    continue
                candidate = energy[u] - D[u][v]
                if med_value[v] != -1:
                    candidate = max(candidate, med_value[v])
                if candidate > energy[v]:
                    energy[v] = candidate
                    updated = True
                    if v == 1:
                        print("Yes")
                        return
        if not updated:
            break
            
    print("No")

if __name__ == '__main__':
    main()