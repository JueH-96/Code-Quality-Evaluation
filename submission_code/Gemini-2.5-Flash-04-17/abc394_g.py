import heapq
import sys

# Using faster input reading
def readline():
    return sys.stdin.readline().rstrip()

def readints():
    return map(int, readline().split())

# Using a large number for infinity
INF = float('inf') # Use float('inf') for safety with large sums

def solve():
    H, W = readints()
    F = []
    for _ in range(H):
        F.append(list(readints()))
    Q = int(readline())

    # Dijkstra's algorithm on states (row, col, type)
    # type 0: floor 1
    # type 1: floor F[row][col]

    # Helper function to calculate min cost |a-k| + |k-b| for k in [L, R]
    # This is the minimum stair cost incurred when transitioning
    # from being on floor 'a' in the current building, taking stairs
    # to floor 'k' (where a walkway is used), moving via walkway (cost 0),
    # arriving at the adjacent building on floor 'k', and then taking stairs
    # to reach floor 'b' in the adjacent building.
    # We minimize this cost over all possible walkway floors 'k' in the range [L, R].
    def get_min_stair_cost_around_walkway(from_floor, to_floor, walkway_min_floor, walkway_max_floor):
        # The function f(k) = |from_floor - k| + |k - to_floor| is convex.
        # Its minimum on the interval [walkway_min_floor, walkway_max_floor]
        # is achieved at max(walkway_min_floor, min(from_floor, to_floor))
        # or min(walkway_max_floor, max(from_floor, to_floor)).
        # If the interval [min(from_floor, to_floor), max(from_floor, to_floor)]
        # overlaps with [walkway_min_floor, walkway_max_floor], the global minimum
        # |from_floor - to_floor| is achievable within the valid range [L, R].

        min_ft = min(from_floor, to_floor)
        max_ft = max(from_floor, to_floor)

        L = walkway_min_floor
        R = walkway_max_floor

        if max_ft < L:
            # The ideal interval [min_ft, max_ft] is entirely below [L, R].
            # The closest point in [L, R] to the ideal interval is L.
            # The function f(k) is decreasing for k < min_ft. So min on [L, R] is at k=L.
            return abs(from_floor - L) + abs(L - to_floor)
        elif min_ft > R:
            # The ideal interval [min_ft, max_ft] is entirely above [L, R].
            # The closest point in [L, R] to the ideal interval is R.
            # The function f(k) is increasing for k > max_ft. So min on [L, R] is at k=R.
            return abs(from_floor - R) + abs(R - to_floor)
        else: # The interval [min_ft, max_ft] overlaps with [L, R]
            # The minimum value of |from_floor - k| + |k - to_floor| over all k is |from_floor - to_floor|.
            # This minimum is achieved for any k in [min_ft, max_ft].
            # Since [min_ft, max_ft] intersects [L,R], there exists a valid k in the intersection
            # that achieves the minimum value |from_floor - to_floor|.
            return abs(from_floor - to_floor)


    for _ in range(Q):
        Ar, Bc, Y, Cr, Dc, Z = readints()

        # Dijkstra state: (cost, row, col, type)
        # type 0: at floor 1
        # type 1: at floor F[row][col]
        # Initialize distances to infinity
        dist = [[([INF] * 2) for _ in range(W)] for _ in range(H)]

        pq = []

        # Initial state: (Ar-1, Bc-1) at floor Y with cost 0.
        # From here, we can use stairs to reach floor 1 or F[Ar-1][Bc-1]
        # within the starting building.

        # Option 1: Use stairs from Y to reach floor 1 at (Ar, Bc)
        initial_cost_to_floor1 = abs(Y - 1)
        dist[Ar-1][Bc-1][0] = initial_cost_to_floor1
        heapq.heappush(pq, (dist[Ar-1][Bc-1][0], Ar-1, Bc-1, 0))

        # Option 2: Use stairs from Y to reach floor F[Ar-1][Bc-1] at (Ar, Bc)
        initial_cost_to_top_floor = abs(Y - F[Ar-1][Bc-1])
        dist[Ar-1][Bc-1][1] = initial_cost_to_top_floor
        heapq.heappush(pq, (dist[Ar-1][Bc-1][1], Ar-1, Bc-1, 1))

        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        while pq:
            cost, r, c, type = heapq.heappop(pq)

            # If we found a shorter path already, skip
            if cost > dist[r][c][type]:
                continue

            # Current building (r, c) (0-indexed)
            curr_F = F[r][c]
            curr_floor = 1 if type == 0 else curr_F

            # 1. Stair move within the building
            next_type = 1 - type
            stair_cost_within = abs(curr_F - 1) # Cost to go from 1 to F or F to 1

            if dist[r][c][next_type] == INF or cost + stair_cost_within < dist[r][c][next_type]:
                 dist[r][c][next_type] = cost + stair_cost_within
                 heapq.heappush(pq, (dist[r][c][next_type], r, c, next_type))

            # 2. Walkway moves to adjacent buildings
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]

                if 0 <= nr < H and 0 <= nc < W:
                    next_F = F[nr][nc]
                    # Walkway at floor k is possible between (r, c) and (nr, nc) if k <= min(F[r][c], F[nr][nc])
                    walkway_max_floor = min(curr_F, next_F)
                    # Walkway is always possible at floor 1 if buildings are tall enough (>0)
                    walkway_min_floor = 1

                    # Transition to (nr, nc) arriving at floor 1 (state type 0)
                    # We transition from curr_floor at (r,c) to floor 1 at (nr,nc)
                    stair_cost_to_reach_floor_1 = get_min_stair_cost_around_walkway(curr_floor, 1, walkway_min_floor, walkway_max_floor)
                    
                    if dist[nr][nc][0] == INF or cost + stair_cost_to_reach_floor_1 < dist[nr][nc][0]:
                        dist[nr][nc][0] = cost + stair_cost_to_reach_floor_1
                        heapq.heappush(pq, (dist[nr][nc][0], nr, nc, 0))

                    # Transition to (nr, nc) arriving at floor next_F (state type 1)
                    # We transition from curr_floor at (r,c) to floor next_F at (nr,nc)
                    stair_cost_to_reach_floor_next_F = get_min_stair_cost_around_walkway(curr_floor, next_F, walkway_min_floor, walkway_max_floor)

                    if dist[nr][nc][1] == INF or cost + stair_cost_to_reach_floor_next_F < dist[nr][nc][1]:
                        dist[nr][nc][1] = cost + stair_cost_to_reach_floor_next_F
                        heapq.heappush(pq, (dist[nr][nc][1], nr, nc, 1))

        # After Dijkstra, find the minimum cost to reach (Cr-1, Dc-1) and then go to Z
        ans = INF

        # Arrive at (Cr-1, Dc-1) at floor 1 (state type 0), then use stairs to Z
        if dist[Cr-1][Dc-1][0] != INF:
            cost_via_floor_1 = dist[Cr-1][Dc-1][0] + abs(1 - Z)
            ans = min(ans, cost_via_floor_1)

        # Arrive at (Cr-1, Dc-1) at floor F[Cr-1][Dc-1] (state type 1), then use stairs to Z
        if dist[Cr-1][Dc-1][1] != INF:
            cost_via_floor_F = dist[Cr-1][Dc-1][1] + abs(F[Cr-1][Dc-1] - Z)
            ans = min(ans, cost_via_floor_F)

        print(ans)

solve()