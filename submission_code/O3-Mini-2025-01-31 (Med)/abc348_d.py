def main():
    import sys
    from collections import deque

    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [list(input().rstrip('
')) for _ in range(H)]
    
    # Find S and T coordinates (0-indexed)
    s_r = s_c = t_r = t_c = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                s_r, s_c = i, j
            if grid[i][j] == 'T':
                t_r, t_c = i, j

    # Read medicines:
    N = int(input())
    # We make a list of nodes. Every node is stored as a tuple:
    # (r, c, med_value, is_goal)
    # For a cell with medicine med_value is the energy value of that medicine.
    # For the target we mark is_goal = True and med_value = None.
    nodes = []  
    # Also, build a dictionary mapping (r,c) to list of node indices.
    cell_nodes = {}
    for idx in range(N):
        R, C, E = map(int, input().split())
        # convert to 0-index
        r, c = R-1, C-1
        node = (r, c, E, False)
        nodes.append(node)
        cell_nodes.setdefault((r, c), []).append(len(nodes)-1)
        
    # Add T as a node (no medicine recharge)
    T_index = len(nodes)
    nodes.append((t_r, t_c, None, True))
    cell_nodes.setdefault((t_r, t_c), []).append(T_index)
    
    # We want to start at S. Since Takahashi starts with 0 energy he must use a medicine
    # in his current cell to get any energy.
    # So if there is no medicine in S (i.e. no node with location S), then it is impossible.
    if (s_r, s_c) not in cell_nodes:
        # no medicine in S -> cannot get started, answer No.
        sys.stdout.write("No")
        return

    # We will use the node (medicine) at S as the starting state.
    # (It may be more than one – but since medicine positions are unique, only one medicine can appear.)
    start_candidates = cell_nodes[(s_r, s_c)]
    # Our initial energy at S will be the maximum among medicine choices at that cell.
    init_energy = -1
    for idx in start_candidates:
        # Only consider nodes that are not T.
        if not nodes[idx][3]:
            init_energy = max(init_energy, nodes[idx][2])
    # If none of the nodes in S cell is a medicine (i.e. they are all the goal, but S != T) then it's impossible.
    if init_energy < 0:
        sys.stdout.write("No")
        return

    # Build distances between nodes.
    # We have at most around 301 nodes.
    n = len(nodes)
    dist = [[None] * n for _ in range(n)]
    
    # We want a quick lookup: for every coordinate, which nodes are here.
    # (cell_nodes already built)

    # helper function: run BFS from (sr, sc) to compute distances to all nodes in our list.
    def bfs_from(sr, sc):
        dmap = [[-1]*W for _ in range(H)]
        dq = deque()
        dq.append((sr, sc))
        dmap[sr][sc] = 0
        # result: mapping from node indices to distance.
        res = {}
        # If (sr, sc) is a target for some node, record:
        if (sr, sc) in cell_nodes:
            for idx in cell_nodes[(sr, sc)]:
                res[idx] = 0

        # Directions
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        while dq:
            r, c = dq.popleft()
            d = dmap[r][c]
            # If we already have distances for all nodes, we can finish early.
            if len(res) == len(cell_nodes_flat):
                # Note: cell_nodes_flat will be computed outside.
                pass  # we do not break because not all nodes are at unique coordinates.
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':  # obstacle
                        continue
                    if dmap[nr][nc] != -1:
                        continue
                    dmap[nr][nc] = d + 1
                    dq.append((nr, nc))
                    if (nr, nc) in cell_nodes:
                        for idx in cell_nodes[(nr, nc)]:
                            # If not recorded before, record.
                            if idx not in res:
                                res[idx] = dmap[nr][nc]
        return res

    # Instead of computing on the fly a set of all nodes,
    # we can just compute BFS for each node.
    for i in range(n):
        r, c, _, _ = nodes[i]
        res = bfs_from(r, c)
        for j, d in res.items():
            dist[i][j] = d

    # Now, we want to “relax” transitions among nodes.
    # Let dp[i] be the maximum energy achievable upon being at node i.
    # When moving from node i to j:
    #   if dp[i] (the energy at i) is at least dist[i][j],
    #   then upon arrival at j, the leftover energy is dp[i]-dist.
    #   Moreover, if j is a medicine node (not T) we can (if beneficial) use it
    #   to set the energy to nodes[j][2].
    # Therefore, candidate = max(dp[i]-dist[i][j], nodes[j][2] (if medicine)).
    dp = [-1] * n

    # For all starting nodes at S that are medicine (not goal), we set dp accordingly.
    for idx in cell_nodes[(s_r, s_c)]:
        if not nodes[idx][3]:  # not T
            # If there are multiple meds at S, take the best energy.
            dp[idx] = max(dp[idx], nodes[idx][2])
    
    # We do a relaxation (like Bellman-Ford / SPFA) over our node graph.
    from collections import deque
    que = deque()
    for i in range(n):
        if dp[i] >= 0:
            que.append(i)
    
    while que:
        u = que.popleft()
        for v in range(n):
            # skip if no path from u to v.
            if dist[u][v] is None:
                continue
            d = dist[u][v]
            # Check if we have enough energy at u to travel the path.
            if dp[u] < d:
                continue
            # Energy left upon arrival
            candidate = dp[u] - d
            # If v is a medicine node we can use the medicine if it gives more energy.
            if not nodes[v][3]:
                med_value = nodes[v][2]
                if med_value is not None:
                    candidate = max(candidate, med_value)
            # For T, we simply keep candidate.
            if candidate > dp[v]:
                dp[v] = candidate
                que.append(v)

    # If we managed to get any energy >= 0 at the target node T, then answer "Yes".
    if dp[T_index] >= 0:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()