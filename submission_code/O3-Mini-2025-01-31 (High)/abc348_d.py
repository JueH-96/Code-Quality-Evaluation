import sys
from collections import deque
import heapq

def main():
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    try:
        H = int(next(it))
        W = int(next(it))
    except StopIteration:
        return
    grid = []
    S_pos = None
    T_pos = None
    for i in range(H):
        line = next(it)
        grid.append(line)
        for j, ch in enumerate(line):
            if ch == 'S':
                S_pos = (i, j)
            elif ch == 'T':
                T_pos = (i, j)
                
    # Read number of medicines
    N = int(next(it))
    med_dict = {}
    for _ in range(N):
        # medicine info: row, col, energy   (1-indexed)
        r = int(next(it)) - 1
        c = int(next(it)) - 1
        E_val = int(next(it))
        med_dict[(r, c)] = E_val

    if S_pos is None or T_pos is None:
        print("No")
        return

    # Build a list of interest nodes.
    # A node is (row, col, med) where med is the energy available if
    # there is medicine on that cell; otherwise we use -1.
    #
    # For the start S, if there is a medicine, that is your only chance to get energy.
    nodes = []
    S_med = med_dict.get(S_pos, -1)
    nodes.append((S_pos[0], S_pos[1], S_med))
    # Include medicine nodes (skip S and T if they occur in the med_dict)
    for pos, val in med_dict.items():
        if pos == S_pos or pos == T_pos:
            continue
        nodes.append((pos[0], pos[1], val))
    # Finally, add goal T (without any medicine effect).
    nodes.append((T_pos[0], T_pos[1], -1))
    
    n_nodes = len(nodes)
    S_index = 0
    T_index = n_nodes - 1

    # Precompute distances between every pair-of-interest-nodes.
    # For each interest node, run a BFS on the grid.
    INF = 10**9
    distances = [[INF]*n_nodes for _ in range(n_nodes)]
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    for i in range(n_nodes):
        sr, sc, _ = nodes[i]
        dist_grid = [[-1]*W for _ in range(H)]
        dq = deque()
        dist_grid[sr][sc] = 0
        dq.append((sr, sc))
        while dq:
            r, c = dq.popleft()
            dcur = dist_grid[r][c]
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if grid[nr][nc] == "#":
                    continue
                if dist_grid[nr][nc] == -1:
                    dist_grid[nr][nc] = dcur + 1
                    dq.append((nr, nc))
        for j in range(n_nodes):
            r, c, _ = nodes[j]
            if dist_grid[r][c] != -1:
                distances[i][j] = dist_grid[r][c]
            else:
                distances[i][j] = INF

    # Now we “relax” our state transitions.
    # Let dp[i] be the maximum energy you can have at interest node i.
    # When going from node u (current energy = dp[u]) to node v spending a cost d = distances[u][v],
    # you arrive with energy (dp[u] - d). On v, if a medicine is present (i.e. nodes[v][2] != -1)
    # you can choose to use it and thus your energy becomes max(dp[u]-d, nodes[v][2]).
    # Note that you can only move if dp[u] >= d.
    dp = [-1] * n_nodes
    # Since you begin at S with 0 energy, only if S has a medicine you can start.
    init_energy = 0
    if S_med != -1:
        init_energy = S_med
    dp[S_index] = init_energy

    # Use a max-heap (storing negative energies) to “propagate” improvements.
    heap = []
    heapq.heappush(heap, (-dp[S_index], S_index))
    
    while heap:
        cur_eng, u = heapq.heappop(heap)
        cur_eng = -cur_eng
        if cur_eng != dp[u]:
            continue
        for v in range(n_nodes):
            if v == u: 
                continue
            cost = distances[u][v]
            if cur_eng < cost:
                continue
            candidate = cur_eng - cost
            med_v = nodes[v][2]
            if med_v != -1:
                if med_v > candidate:
                    candidate = med_v
            if candidate > dp[v]:
                dp[v] = candidate
                heapq.heappush(heap, (-candidate, v))
                
    # If we could reach the goal T (even with 0 leftover energy) then answer Yes.
    if dp[T_index] >= 0:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()