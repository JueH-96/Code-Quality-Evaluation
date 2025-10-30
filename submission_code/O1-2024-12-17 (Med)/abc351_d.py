def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    S = input_data[2:]

    # Directions for adjacency (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Step 1: Identify which cells are magnets and which are not
    is_magnet = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if S[i][j] == '#':
                is_magnet[i][j] = True

    # Step 2: Determine "stuck" vs "free" for each '.' cell
    # A cell is "stuck" if it is '.' and at least one adjacent cell is a magnet
    # Otherwise, if it is '.', it's "free".
    # If it is '#', we ignore it in movement calculations.
    is_stuck = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if not is_magnet[i][j]:  # it's a '.' cell
                # Check neighbors for a magnet
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if is_magnet[ni][nj]:
                            is_stuck[i][j] = True
                            break

    # Step 3: Find connected components among the "free" cells.
    # A "free" cell is '.' and not stuck (no adjacent magnets).
    comp_of = [[-1]*W for _ in range(H)]
    comp_size = []
    comp_adj_stuck_count = []

    def bfs_component(start_i, start_j, comp_id):
        queue = deque()
        queue.append((start_i, start_j))
        comp_of[start_i][start_j] = comp_id
        size_count = 0
        stuck_neighbors = set()

        while queue:
            r, c = queue.popleft()
            size_count += 1
            # Check all neighbors
            for di, dj in directions:
                nr, nc = r + di, c + dj
                if 0 <= nr < H and 0 <= nc < W:
                    if not is_magnet[nr][nc]:
                        if is_stuck[nr][nc]:
                            # It's a stuck cell, so we can't expand into it,
                            # but it is reachable from this free cell.
                            stuck_neighbors.add((nr, nc))
                        else:
                            # It's a free cell
                            if comp_of[nr][nc] == -1:
                                comp_of[nr][nc] = comp_id
                                queue.append((nr, nc))

        comp_size[comp_id] = size_count
        comp_adj_stuck_count[comp_id] = len(stuck_neighbors)

    # Perform BFS for each unvisited free cell
    comp_id_counter = 0
    for i in range(H):
        for j in range(W):
            # If this cell is a free cell and not yet visited, find its component
            if not is_magnet[i][j] and not is_stuck[i][j] and comp_of[i][j] == -1:
                comp_size.append(0)
                comp_adj_stuck_count.append(0)
                bfs_component(i, j, comp_id_counter)
                comp_id_counter += 1

    # Step 4: Compute degrees of freedom and find the maximum
    max_dof = 0

    for i in range(H):
        for j in range(W):
            if is_magnet[i][j]:
                # Magnet cells are not considered for answer
                continue
            if is_stuck[i][j]:
                # Degree of freedom for a stuck cell is 1
                dof = 1
            else:
                # It's a free cell, belongs to some component
                cid = comp_of[i][j]
                dof = comp_size[cid] + comp_adj_stuck_count[cid]
            if dof > max_dof:
                max_dof = dof

    print(max_dof)

# Don't forget to call main()!
if __name__ == "__main__":
    main()