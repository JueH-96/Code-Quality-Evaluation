def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    S = input_data[2:]

    # Directions for up, right, down, left
    directions = [(-1,0),(0,1),(1,0),(0,-1)]

    # Grid to mark:
    #   2 = magnet
    #   1 = blocked (empty but has at least one adjacent magnet)
    #   0 = unblocked (empty with no adjacent magnets)
    grid_type = [[-1]*W for _ in range(H)]

    # Pre-check magnets
    for i in range(H):
        for j in range(W):
            if S[i][j] == '#':
                grid_type[i][j] = 2  # magnet

    # Determine blocked/unblocked for '.' cells
    for i in range(H):
        for j in range(W):
            if grid_type[i][j] == 2:
                continue  # magnet
            # Check if (i,j) has a magnet neighbor
            is_blocked = False
            for di, dj in directions:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    if S[ni][nj] == '#':
                        is_blocked = True
                        break
            grid_type[i][j] = 1 if is_blocked else 0

    # We will find connected components among unblocked cells (grid_type==0)
    comp_id = [[-1]*W for _ in range(H)]
    comp_sizes = []
    comp_blocked_neighbors = []
    current_comp = 0

    def bfs_unblocked(sr, sc):
        """Perform BFS starting from (sr, sc) over unblocked cells.
           Returns (component_size, set_of_blocked_neighbors)."""
        queue = deque()
        queue.append((sr, sc))
        comp_id[sr][sc] = current_comp
        cells_in_comp = 1
        blocked_set = set()

        while queue:
            r, c = queue.popleft()
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid_type[nr][nc] == 0:
                        # unblocked neighbor
                        if comp_id[nr][nc] == -1:
                            comp_id[nr][nc] = current_comp
                            queue.append((nr, nc))
                            cells_in_comp += 1
                    elif grid_type[nr][nc] == 1:
                        # blocked neighbor (empty but adjacent to magnet)
                        # We'll just record it in the set
                        blocked_set.add((nr, nc))
                    # if magnet (2), we ignore it altogether
        return cells_in_comp, blocked_set

    # Build connected components for unblocked cells
    for i in range(H):
        for j in range(W):
            if grid_type[i][j] == 0 and comp_id[i][j] == -1:
                size_comp, blocked_nbrs = bfs_unblocked(i, j)
                comp_sizes.append(size_comp)
                comp_blocked_neighbors.append(len(blocked_nbrs))
                current_comp += 1

    # Now compute maximum degree of freedom
    max_dof = 0

    # Case 1: blocked cells => dof = 1
    # Case 2: unblocked cell => dof = comp_size + number_of_blocked_neighbors
    # (since all blocked neighbors of that component are reachable)
    for i in range(H):
        for j in range(W):
            if grid_type[i][j] == 2:
                # magnet, skip
                continue
            elif grid_type[i][j] == 1:
                # blocked cell
                max_dof = max(max_dof, 1)
            else:
                # unblocked cell
                cid = comp_id[i][j]
                dof = comp_sizes[cid] + comp_blocked_neighbors[cid]
                if dof > max_dof:
                    max_dof = dof

    print(max_dof)

# Call main() at the end
if __name__ == "__main__":
    main()