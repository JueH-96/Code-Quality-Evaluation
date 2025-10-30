import heapq
import sys

# Increase recursion depth if needed, although Dijkstra is iterative.
# Set to a sufficiently large value if potential issues with recursive helper functions arise,
# but this specific Dijkstra implementation is iterative.
sys.setrecursionlimit(2000)

def solve():
    # Read H, W
    H, W = map(int, sys.stdin.readline().split())

    # Read building floors
    F = []
    for _ in range(H):
        F.append(list(map(int, sys.stdin.readline().split())))

    # Read number of queries
    Q = int(sys.stdin.readline())

    # Direction vectors for adjacent blocks (up, down, left, right)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # Process each query
    for _ in range(Q):
        # Read query parameters (start row, col, floor; end row, col, floor)
        Ar, Bc, Y, Cr, Dc, Z = map(int, sys.stdin.readline().split())
        # Convert to 0-indexed coordinates
        Ar -= 1 
        Bc -= 1 
        Cr -= 1 
        Dc -= 1 

        # Dijkstra implementation
        # State: (cost, row, col, type)
        # type 0: represents reaching floor 1 in block (row, col)
        # type 1: represents reaching floor F[row][col] (the top floor) in block (row, col)
        
        # Initialize distance array with infinity for all states (H x W x 2)
        # Use float('inf') for initial large distance values
        dist = [[ [float('inf')] * 2 for _ in range(W)] for _ in range(H)]
        
        # Priority queue for Dijkstra, stores tuples (cost, row, col, type)
        # heapq maintains a min-heap
        pq = []

        # Initial state: starting at (Ar, Bc) on floor Y.
        # The cost is the total stair moves from Y to either floor 1 or floor F[Ar][Bc] in the starting block.
        
        # Cost to reach floor 1 in (Ar, Bc) from initial floor Y
        dist[Ar][Bc][0] = abs(Y - 1)
        heapq.heappush(pq, (dist[Ar][Bc][0], Ar, Bc, 0))

        # Cost to reach floor F[Ar][Bc] in (Ar, Bc) from initial floor Y
        dist[Ar][Bc][1] = abs(Y - F[Ar][Bc])
        heapq.heappush(pq, (dist[Ar][Bc][1], Ar, Bc, 1))

        # Run Dijkstra
        while pq:
            # Pop the state with the smallest cost from the priority queue
            cost, r, c, type = heapq.heappop(pq)

            # If we found a shorter path to this state already, ignore this one
            if cost > dist[r][c][type]:
                continue

            current_F = F[r][c] # Floor count of the current building (r, c)

            # --- Transitions within the same building (r, c) ---
            # We can move between floor 1 and floor F[r][c] using stairs
            if type == 0: # currently at floor 1 in (r, c) with cost 'cost'
                # Move to floor F[r][c] within the same building
                # Stair cost is the absolute difference in floor levels
                new_cost = cost + abs(current_F - 1)
                # If this new path is shorter, update distance and push to PQ
                if new_cost < dist[r][c][1]:
                    dist[r][c][1] = new_cost
                    heapq.heappush(pq, (new_cost, r, c, 1))
            else: # type == 1, currently at floor F[r][c] in (r, c) with cost 'cost'
                # Move to floor 1 within the same building
                # Stair cost is the absolute difference in floor levels
                new_cost = cost + abs(current_F - 1)
                # If this new path is shorter, update distance and push to PQ
                if new_cost < dist[r][c][0]:
                    dist[r][c][0] = new_cost
                    heapq.heappush(pq, (new_cost, r, c, 0))

            # --- Transitions to adjacent buildings (nr, nc) ---
            # Check all 4 adjacent blocks
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]

                # Check if neighbor is within grid bounds (0 to H-1, 0 to W-1)
                if 0 <= nr < H and 0 <= nc < W:
                    neighbor_F = F[nr][nc] # Floor count of the neighbor building (nr, nc)

                    if type == 0: # currently at floor 1 in (r, c) with cost 'cost'
                        # Option A: Move to (nr, nc) floor 1 (type 0)
                        # This can be done by using a walkway at floor 1.
                        # Path: (r,c) f1 --walkway(f1)--> (nr,nc) f1
                        # Walkway at floor X from (i,j) to (i',j') is possible if F[i',j'] >= X.
                        # Here X=1. Need F[nr][nc] >= 1. Since all F values are >= 1, this is always possible.
                        # Stair cost for walkway is 0.
                        # Total cost: current_cost + 0
                        new_cost_0 = cost + 0
                        if new_cost_0 < dist[nr][nc][0]:
                            dist[nr][nc][0] = new_cost_0
                            heapq.heappush(pq, (new_cost_0, nr, nc, 0))

                        # Option B: Move to (nr, nc) floor F[nr][nc] (type 1)
                        # A possible path is: (r,c) f1 --walkway(f1)--> (nr,nc) f1 --stairs--> (nr,nc) fF[nr][nc]
                        # Walkway at floor 1 is always possible as discussed above. Cost 0.
                        # Stairs cost in (nr,nc) from floor 1 to floor F[nr][nc] is abs(1 - F[nr][nc]).
                        # Total cost: current_cost + 0 + abs(1 - F[nr][nc])
                        new_cost_1 = cost + abs(1 - neighbor_F)
                        if new_cost_1 < dist[nr][nc][1]:
                            dist[nr][nc][1] = new_cost_1
                            heapq.heappush(pq, (new_cost_1, nr, nc, 1))

                    else: # type == 1, currently at floor F[r][c] in (r, c) with cost 'cost'
                        # Option A: Move to (nr, nc) floor 1 (type 0)
                        # A possible path is: (r,c) fF[r][c] --stairs--> (r,c) f1 --walkway(f1)--> (nr,nc) f1
                        # Stairs cost in (r,c) from floor F[r][c] to floor 1 is abs(F[r][c] - 1).
                        # Walkway at floor 1 is always possible. Cost 0.
                        # Total cost: current_cost + abs(F[r][c] - 1) + 0
                        new_cost_0 = cost + abs(current_F - 1)
                        if new_cost_0 < dist[nr][nc][0]:
                            dist[nr][nc][0] = new_cost_0
                            heapq.heappush(pq, (new_cost_0, nr, nc, 0))

                        # Option B: Move to (nr, nc) floor F[nr][nc] (type 1)
                        # A possible path is: (r,c) fF[r][c] --stairs--> (r,c) fX --walkway(fX)--> (nr,nc) fX --stairs--> (nr,nc) fF[nr][nc]
                        # To minimize stair cost |F[r][c] - X| + |X - F[nr][nc]|, X should be between F[r][c] and F[nr][nc].
                        # The walkway at floor X needs F[r][c] >= X and F[nr][nc] >= X.
                        # This requires X <= min(F[r][c], F[nr][nc]).
                        # The optimal X in the valid range [1, min(F[r][c], F[nr][nc])] is X = min(F[r][c], F[nr][nc]).
                        # The minimum stair cost for this transition is abs(F[r][c] - min(F[r][c], F[nr][nc])) + abs(min(F[r][c], F[nr][nc]) - F[nr][nc]) = abs(F[r][c] - F[nr][nc]).
                        # Total cost: current_cost + abs(F[r][c] - F[nr][nc])
                        new_cost_1 = cost + abs(current_F - neighbor_F)
                        if new_cost_1 < dist[nr][nc][1]:
                            dist[nr][nc][1] = new_cost_1
                            heapq.heappush(pq, (new_cost_1, nr, nc, 1))

        # After Dijkstra has explored all reachable states, find the minimum cost to reach the target (Cr, Dc) floor Z
        # We can reach target floor Z in block (Cr, Dc) by taking stairs from either floor 1 or floor F[Cr][Dc]
        
        # Option 1: Reach (Cr, Dc) floor 1 with cost dist[Cr][Dc][0], then stairs to floor Z
        ans1 = dist[Cr][Dc][0] + abs(Z - 1)
        
        # Option 2: Reach (Cr, Dc) floor F[Cr][Dc] with cost dist[Cr][Dc][1], then stairs to floor Z
        ans2 = dist[Cr][Dc][1] + abs(Z - F[Cr][Dc])

        # The overall minimum cost is the minimum of these two options
        sys.stdout.write(str(min(ans1, ans2)) + "
")

# Main execution block
if __name__ == "__main__":
    # Speed up input/output operations
    sys.stdin.readline = sys.stdin.buffer.readline
    sys.stdout.write = sys.stdout.buffer.write
    
    solve()