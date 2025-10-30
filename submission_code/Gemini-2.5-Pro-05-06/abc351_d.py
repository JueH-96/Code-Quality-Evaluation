import collections

def solve():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]

    mobile = [[False for _ in range(W)] for _ in range(H)]

    # Directions for neighbors: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Step 1: Precompute mobile[r][c]
    for r in range(H):
        for c in range(W):
            if S[r][c] == '#':
                continue  # Magnets are not mobile

            is_cell_mobile = True
            for dr_val, dc_val in directions:
                nr, nc = r + dr_val, c + dc_val
                # Check if neighbor is within bounds
                if 0 <= nr < H and 0 <= nc < W:
                    if S[nr][nc] == '#':
                        is_cell_mobile = False
                        break
            if is_cell_mobile:
                mobile[r][c] = True

    # Problem guarantees at least one empty cell.
    # Its degree of freedom is at least 1 (it can reach itself).
    max_overall_degree_freedom = 1
    
    # visited_component_bfs[r][c] is true if (r,c) has been part of a
    # mobile component that was already processed.
    visited_component_bfs = [[False for _ in range(W)] for _ in range(H)]

    # Step 2 & 3: Find components of mobile cells and calculate max degree of freedom
    for r_start_comp in range(H):
        for c_start_comp in range(W):
            # If this cell is mobile and its component hasn't been processed yet:
            if mobile[r_start_comp][c_start_comp] and \
               not visited_component_bfs[r_start_comp][c_start_comp]:
                
                # Start BFS to find all cells in this connected component of mobile cells
                current_component_cells = set()
                q_comp = collections.deque()

                q_comp.append((r_start_comp, c_start_comp))
                visited_component_bfs[r_start_comp][c_start_comp] = True
                
                while q_comp:
                    curr_r, curr_c = q_comp.popleft()
                    current_component_cells.add((curr_r, curr_c))
                    
                    for dr_val, dc_val in directions:
                        adj_r, adj_c = curr_r + dr_val, curr_c + dc_val
                        if 0 <= adj_r < H and 0 <= adj_c < W:
                            # If neighbor is mobile and not yet part of a processed component
                            if mobile[adj_r][adj_c] and not visited_component_bfs[adj_r][adj_c]:
                                visited_component_bfs[adj_r][adj_c] = True
                                q_comp.append((adj_r, adj_c))
                
                # Component found: current_component_cells
                # Calculate degree of freedom for starting in this component.
                # It's the size of the union of component cells and their direct neighbors.
                all_reachable_nodes_for_component = set()
                for comp_r, comp_c in current_component_cells:
                    all_reachable_nodes_for_component.add((comp_r, comp_c))
                    
                    # Add its neighbors. Since (comp_r, comp_c) is mobile, 
                    # all its neighbors are guaranteed to be empty.
                    for dr_val, dc_val in directions:
                        adj_r, adj_c = comp_r + dr_val, comp_c + dc_val
                        if 0 <= adj_r < H and 0 <= adj_c < W: # Check bounds for neighbor
                            all_reachable_nodes_for_component.add((adj_r, adj_c))
                
                degree_of_freedom_for_this_component = len(all_reachable_nodes_for_component)
                if degree_of_freedom_for_this_component > max_overall_degree_freedom:
                    max_overall_degree_freedom = degree_of_freedom_for_this_component
    
    # Step 4: Print the result
    print(max_overall_degree_freedom)

solve()