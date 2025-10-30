def main():
    import sys
    from collections import deque
    
    data = sys.stdin.read().strip().split()
    H, W = map(int, data[:2])
    S = data[2:]
    
    # 1) Identify which cells have magnets
    magnet = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if S[i][j] == '#':
                magnet[i][j] = True
    
    # 2) Identify "edge" cells: empty cells adjacent to at least one magnet
    edge = [[False]*W for _ in range(H)]
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    for i in range(H):
        for j in range(W):
            if not magnet[i][j]:  # empty cell
                for dr, dc in directions:
                    nr, nc = i+dr, j+dc
                    if 0 <= nr < H and 0 <= nc < W and magnet[nr][nc]:
                        edge[i][j] = True
                        break
    
    # 3) "Free" cells: empty cells NOT adjacent to any magnet
    free = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if not magnet[i][j] and not edge[i][j]:
                free[i][j] = True
    
    # 4) For each connected component of free cells, compute:
    #    degree_of_freedom = (#free_cells_in_component) + (#edge_cells_adjacent_to_that_component)
    visited = [[False]*W for _ in range(H)]
    max_degree = 0
    
    for i in range(H):
        for j in range(W):
            # BFS/DFS only on unvisited free cells
            if free[i][j] and not visited[i][j]:
                queue = deque([(i,j)])
                visited[i][j] = True
                comp_size = 0
                edge_neighbors = set()
                
                while queue:
                    r, c = queue.popleft()
                    comp_size += 1
                    # Check 4-directional neighbors
                    for dr, dc in directions:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if free[nr][nc] and not visited[nr][nc]:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                            elif edge[nr][nc]:
                                # Count each edge cell once
                                edge_neighbors.add((nr, nc))
                
                # Degree of freedom for this free-region
                max_degree = max(max_degree, comp_size + len(edge_neighbors))
    
    # 5) Cells that are "edge" (adjacent to a magnet) have degree 1 if started from there,
    #    so answer is max(1, best we found among free cells).
    print(max(1, max_degree))

# Do not forget to call main():
if __name__ == "__main__":
    main()