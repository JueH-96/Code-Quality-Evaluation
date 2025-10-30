import heapq
import sys

def solve():
    # Read H and W
    H, W = map(int, sys.stdin.readline().split())

    # Read building floors
    F = []
    for _ in range(H):
        F.append(list(map(int, sys.stdin.readline().split())))

    # Read number of queries
    Q = int(sys.stdin.readline())

    # Directions for cardinal adjacency (up, down, left, right)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    results = []

    for query_idx in range(Q):
        A_i, B_i, Y_i, C_i, D_i, Z_i = map(int, sys.stdin.readline().split())
        
        # Adjust to 0-indexed coordinates
        r_start, c_start = A_i - 1, B_i - 1
        r_end, c_end = C_i - 1, D_i - 1
        y_start, z_end = Y_i, Z_i

        # dist[r][c][type_idx] stores the minimum stair cost to reach building (r,c)
        # type_idx = 0 means at floor 1
        # type_idx = 1 means at floor F[r][c] (top floor of building (r,c))
        
        # Initialize distances to infinity using a 3D list for efficient access
        dist = [[[float('inf')] * 2 for _ in range(W)] for _ in range(H)]
        
        # Priority queue for Dijkstra: (current_total_cost, r, c, type_idx)
        pq = []
        
        # Initial states based on Y_i (start floor of the query)
        
        # Calculate cost to reach (r_start, c_start) at floor 1 from Y_i
        cost_to_1_from_Y = abs(y_start - 1)
        if cost_to_1_from_Y < dist[r_start][c_start][0]:
            dist[r_start][c_start][0] = cost_to_1_from_Y
            heapq.heappush(pq, (cost_to_1_from_Y, r_start, c_start, 0))

        # Calculate cost to reach (r_start, c_start) at its top floor F[r_start][c_start] from Y_i
        cost_to_F_start_from_Y = abs(y_start - F[r_start][c_start])
        if cost_to_F_start_from_Y < dist[r_start][c_start][1]:
            dist[r_start][c_start][1] = cost_to_F_start_from_Y
            heapq.heappush(pq, (cost_to_F_start_from_Y, r_start, c_start, 1))
            
        while pq:
            d, r, c, type_idx = heapq.heappop(pq)
            
            # If we found a shorter path previously, skip this one
            if d > dist[r][c][type_idx]:
                continue
                
            # Determine the current actual floor based on type_idx
            curr_floor = 1 if type_idx == 0 else F[r][c]
            
            # --- Transitions within the current building (stairs) ---
            # Option 1: Move to floor 1 in current building (r,c)
            new_d_to_1 = d + abs(curr_floor - 1)
            if new_d_to_1 < dist[r][c][0]:
                dist[r][c][0] = new_d_to_1
                heapq.heappush(pq, (new_d_to_1, r, c, 0))
            
            # Option 2: Move to top floor F[r][c] in current building (r,c)
            new_d_to_F_rc = d + abs(curr_floor - F[r][c])
            if new_d_to_F_rc < dist[r][c][1]:
                dist[r][c][1] = new_d_to_F_rc
                heapq.heappush(pq, (new_d_to_F_rc, r, c, 1))
                
            # --- Walkway to adjacent buildings ---
            for i in range(4): # Iterate over 4 cardinal directions (up, down, left, right)
                nr, nc = r + dr[i], c + dc[i]
                
                # Check if the adjacent coordinates are within grid boundaries
                if 0 <= nr < H and 0 <= nc < W:
                    # A walkway is possible from current building at curr_floor to an adjacent building (nr,nc)
                    # at the same floor curr_floor, provided curr_floor <= F[nr][nc]
                    if curr_floor <= F[nr][nc]:
                        # From (r,c) at curr_floor to (nr,nc) using walkway (cost 0 stair uses)
                        
                        # Option 1: Arrive at (nr,nc) and then move to its 1st floor
                        # Cost: d (to get to (r,c) at curr_floor) + abs(curr_floor - 1) (stairs at (nr,nc))
                        new_d_adj_to_1 = d + abs(curr_floor - 1)
                        if new_d_adj_to_1 < dist[nr][nc][0]:
                            dist[nr][nc][0] = new_d_adj_to_1
                            heapq.heappush(pq, (new_d_adj_to_1, nr, nc, 0))
                        
                        # Option 2: Arrive at (nr,nc) and then move to its top floor F[nr][nc]
                        # Cost: d (to get to (r,c) at curr_floor) + abs(curr_floor - F[nr][nc]) (stairs at (nr,nc))
                        new_d_adj_to_F_nrc = d + abs(curr_floor - F[nr][nc])
                        if new_d_adj_to_F_nrc < dist[nr][nc][1]:
                            dist[nr][nc][1] = new_d_adj_to_F_nrc
                            heapq.heappush(pq, (new_d_adj_to_F_nrc, nr, nc, 1))

        # After Dijkstra finishes for this query, calculate the final answer
        min_total_stairs = float('inf')
        
        # Consider the path ending at (r_end, c_end) having reached its 1st floor
        if dist[r_end][c_end][0] != float('inf'):
            min_total_stairs = min(min_total_stairs, dist[r_end][c_end][0] + abs(1 - z_end))
            
        # Consider the path ending at (r_end, c_end) having reached its top floor F[r_end][c_end]
        if dist[r_end][c_end][1] != float('inf'):
            min_total_stairs = min(min_total_stairs, dist[r_end][c_end][1] + abs(F[r_end][c_end] - z_end))
            
        results.append(min_total_stairs)

    # Print all collected results, each on a new line
    for res in results:
        sys.stdout.write(str(res) + '
')

solve()