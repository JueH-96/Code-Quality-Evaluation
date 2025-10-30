import collections
import sys

def solve():
    """
    Solves the Energy Drink problem by modeling it as a graph reachability problem.
    """
    # Use sys.stdin.readline for faster I/O in competitive programming
    input = sys.stdin.readline

    # --- 1. Parse Input and Initial Checks ---
    try:
        line = input()
        if not line: return
        H, W = map(int, line.split())
        grid = [input().strip() for _ in range(H)]
        N = int(input())
        meds = []
        if N > 0:
            for _ in range(N):
                r, c, e = map(int, input().split())
                meds.append((r - 1, c - 1, e)) # Convert to 0-based indexing
    except (IOError, ValueError):
        return

    start_pos, target_pos = None, None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start_pos = (r, c)
            elif grid[r][c] == 'T':
                target_pos = (r, c)
    
    if start_pos == target_pos:
        print("Yes")
        return

    med_loc_to_info = {(r, c): (i, e) for i, (r, c, e) in enumerate(meds)}

    if start_pos not in med_loc_to_info:
        print("No")
        return

    # --- 2. Pre-computation of Distances ---
    def bfs_grid(start_r, start_c):
        q = collections.deque([(start_r, start_c, 0)])
        distances = {(start_r, start_c): 0}
        
        while q:
            r, c, d = q.popleft()
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and (nr, nc) not in distances:
                    distances[(nr, nc)] = d + 1
                    q.append((nr, nc, d + 1))
        return distances

    dist_from_med = []
    if N > 0:
        for i in range(N):
            med_r, med_c, _ = meds[i]
            dist_from_med.append(bfs_grid(med_r, med_c))
    
    # --- 3. Build a "Medicine Graph" ---
    adj = [[] for _ in range(N)]
    for i in range(N):
        energy_i = meds[i][2]
        for j in range(N):
            if i == j:
                continue
            
            loc_j = (meds[j][0], meds[j][1])
            dist_ij = dist_from_med[i].get(loc_j, float('inf'))
            
            if energy_i >= dist_ij:
                adj[i].append(j)

    # --- 4. & 5. Check for Reachability and Final Output ---
    start_med_idx = med_loc_to_info[start_pos][0]
    q_med_graph = collections.deque([start_med_idx])
    visited_meds = {start_med_idx}

    energy_start = meds[start_med_idx][2]
    dist_to_target = dist_from_med[start_med_idx].get(target_pos, float('inf'))
    if energy_start >= dist_to_target:
        print("Yes")
        return
    
    while q_med_graph:
        u = q_med_graph.popleft()
        
        for v in adj[u]:
            if v not in visited_meds:
                visited_meds.add(v)
                
                energy_v = meds[v][2]
                dist_to_target_from_v = dist_from_med[v].get(target_pos, float('inf'))
                
                if energy_v >= dist_to_target_from_v:
                    print("Yes")
                    return
                
                q_med_graph.append(v)
            
    print("No")

if __name__ == "__main__":
    solve()