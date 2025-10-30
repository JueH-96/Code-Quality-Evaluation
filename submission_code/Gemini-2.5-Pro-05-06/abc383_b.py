def solve():
    H, W, D = map(int, input().split())
    S_grid = []
    for _ in range(H):
        S_grid.append(input())

    floor_cells = []
    for r_idx in range(H):
        for c_idx in range(W):
            if S_grid[r_idx][c_idx] == '.':
                floor_cells.append((r_idx, c_idx))

    num_floor_cells = len(floor_cells)
    
    # Constraint: "There are at least two floor cells."
    # So num_floor_cells >= 2 is guaranteed.

    max_humidified_count = 0

    # Iterate over all possible pairs of distinct floor cells (p1_coords, p2_coords)
    # to place humidifiers.
    # p1_coords will be floor_cells[i]
    # p2_coords will be floor_cells[j]
    for i in range(num_floor_cells):
        for j in range(i + 1, num_floor_cells): # j > i ensures p1 and p2 are distinct and avoids duplicate pairs
            
            p1_coords = floor_cells[i]
            p2_coords = floor_cells[j]
            
            r1, c1 = p1_coords
            r2, c2 = p2_coords
            
            current_humidified_count = 0
            # Iterate over all floor cells (pf_coords) to check if they are humidified
            # by the chosen pair of humidifiers (p1_coords, p2_coords).
            for k in range(num_floor_cells):
                pf_coords = floor_cells[k] 
                rf, cf = pf_coords
                
                # Manhattan distance from pf_coords to p1_coords
                dist_to_p1 = abs(rf - r1) + abs(cf - c1)
                # Manhattan distance from pf_coords to p2_coords
                dist_to_p2 = abs(rf - r2) + abs(cf - c2)
                
                if dist_to_p1 <= D or dist_to_p2 <= D:
                    current_humidified_count += 1
            
            # Update the maximum count found so far
            if current_humidified_count > max_humidified_count:
                max_humidified_count = current_humidified_count
            # Alternatively, and more idiomatically:
            # max_humidified_count = max(max_humidified_count, current_humidified_count)
            
    print(max_humidified_count)

if __name__ == '__main__':
    solve()