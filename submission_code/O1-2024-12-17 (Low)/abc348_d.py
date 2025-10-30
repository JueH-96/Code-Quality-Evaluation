from collections import deque

def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    H, W = map(int, input_data[:2])
    idx = 2
    grid = input_data[idx:idx+H]
    idx += H

    # Locate S and T
    s_r = s_c = -1
    t_r = t_c = -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                s_r, s_c = r, c
            elif grid[r][c] == 'T':
                t_r, t_c = r, c

    N = int(input_data[idx])
    idx += 1

    medicines = []
    # We'll store (row-1, col-1, E)
    # also note if this overlaps with S or T
    for _ in range(N):
        rr = int(input_data[idx]); cc = int(input_data[idx+1]); ee = int(input_data[idx+2])
        idx += 3
        medicines.append((rr-1, cc-1, ee))

    # We will create a list of "special nodes":
    # node 0 -> S
    # node 1..N -> each medicine
    # node N+1 -> T
    # We'll compute BFS-distances between these nodes if reachable.
    # Then we'll do a relaxation approach to see if we can end up with a non-negative best[T].
    
    # Build the list of special nodes
    # Each element: (r, c, energy_if_medicine_else_None)
    # S will have None, T will have None, medicines have E_i
    special_nodes = []
    # 0 = S
    special_nodes.append((s_r, s_c, None))
    # 1..N = medicines
    for i in range(N):
        special_nodes.append((medicines[i][0], medicines[i][1], medicines[i][2]))
    # N+1 = T
    special_nodes.append((t_r, t_c, None))

    node_count = N + 2
    # dist[i][j] = distance from node i to node j on the grid (or inf if not reachable)
    INF = 10**15
    dist = [[INF]*(node_count) for _ in range(node_count)]

    # Directions
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Precompute BFS from each special node
    # BFS in the grid of size H*W
    def bfs_from(r0, c0):
        # returns array of size node_count with distances;
        # or INF if not reachable
        dist_map = [[-1]*W for _ in range(H)]
        dq = deque()
        dist_map[r0][c0] = 0
        dq.append((r0, c0))
        while dq:
            rr, cc = dq.popleft()
            for dr, dc in dirs:
                nr, nc = rr+dr, cc+dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] != '#' and dist_map[nr][nc] < 0:
                        dist_map[nr][nc] = dist_map[rr][cc] + 1
                        dq.append((nr,nc))
        # now read off distances to each special node
        result = []
        for (rx, cx, _) in special_nodes:
            result.append(dist_map[rx][cx] if rx>=0 and cx>=0 else -1)
        return result

    # Perform BFS from each special node i
    for i in range(node_count):
        (r0, c0, _) = special_nodes[i]
        bfs_result = bfs_from(r0, c0)
        for j in range(node_count):
            dval = bfs_result[j]
            if dval >= 0:
                dist[i][j] = dval

    # Now we do the relaxation on the condensed graph of node_count nodes:
    # best[i] = the maximum energy we can have upon "arriving" at node i.
    # At S, we might immediately pick up medicine if it exists there.
    # So if there's a medicine on S, best[0] = that E. Otherwise 0.
    # Then we repeatedly update:
    #   if best[i] >= dist[i][j], we can go from i -> j (this uses dist[i][j] energy),
    #   so upon arrival at j, we have new_candidate = best[i] - dist[i][j].
    #   Then we can pick up the medicine at j (if any) to possibly become E_j.
    #   So new_energy_at_j = max(new_candidate, E_j-if-j-has-medicine).
    #   If new_energy_at_j > best[j], we update best[j].
    # We stop when no updates happen or we find best[T] != -1 (meaning we have some energy there).

    best_energy = [-1]*(node_count)

    # Check if S has a medicine (node=0 has special_nodes[0] -> None? or we search in medicines)
    # Actually, we already included all medicines in the special_nodes list from index=1..N.
    # If a medicine is physically at S's cell, we will find it at some index k in 1..N.
    # But that doesn't give us best_energy[0] immediately. We must see if dist(0,k)=0, then we can pick it up.
    # So let's start best_energy[S]=0 because we are physically at S with energy=0.
    best_energy[0] = 0

    changed = True
    while changed:
        changed = False
        for i in range(node_count):
            if best_energy[i] < 0:  # not reachable so far
                continue
            e_i = best_energy[i]
            # try to move from i to j
            for j in range(node_count):
                dcost = dist[i][j]
                if dcost == INF:  # not reachable
                    continue
                if e_i >= dcost:
                    # we can move to j
                    new_candidate = e_i - dcost
                    # possibly pick up medicine at j
                    # special_nodes[j] = (r_j, c_j, E_j) or None if j is S or T
                    _, _, e_med = special_nodes[j]
                    if e_med is not None:
                        # after picking up that medicine if we want
                        # we can choose either to keep new_candidate or pick e_med
                        # so final is max(new_candidate, e_med)
                        new_candidate = max(new_candidate, e_med)
                    if new_candidate > best_energy[j]:
                        best_energy[j] = new_candidate
                        changed = True
        # if best_energy for T (node N+1) is updated to >= 0, we already can reach T.
        if best_energy[node_count-1] >= 0:
            break

    if best_energy[node_count-1] >= 0:
        print("Yes")
    else:
        print("No")

# Call main() at the end
if __name__ == "__main__":
    main()