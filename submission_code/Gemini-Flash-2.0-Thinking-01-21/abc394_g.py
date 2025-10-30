import heapq
import sys

# Set recursion depth for large inputs if needed
# sys.setrecursionlimit(2000)

def solve():
    H, W = map(int, sys.stdin.readline().split())
    F = []
    for _ in range(H):
        F.append(list(map(int, sys.stdin.readline().split())))

    Q = int(sys.stdin.readline())
    queries = []
    for _ in range(Q):
        queries.append(list(map(int, sys.stdin.readline().split())))

    def min_stairs(start_r, start_c, start_y, end_r, end_c, end_z):
        # Use 0-indexed coordinates
        start_r -= 1
        start_c -= 1
        end_r -= 1
        end_c -= 1

        # dist[r][c][0]: min cost to reach (r, c) and then adjust to floor 1 using stairs
        # dist[r][c][1]: min cost to reach (r, c) and then adjust to floor F[r][c] using stairs
        # The cost includes all stair moves up to the point of reaching (r,c)
        # AND the final stair adjustment to floor 1 or F[r][c].
        # This represents the minimum total stair cost to reach building (r,c) and specifically be on floor 1 or F[r][c].
        
        dist = [[[-1, -1] for _ in range(W)] for _ in range(H)]

        pq = []

        # Initial state: Start at (start_r, start_c) at floor start_y
        # Cost to reach (start_r, start_c) and go to floor 1 is |start_y - 1|
        dist[start_r][start_c][0] = abs(start_y - 1)
        heapq.heappush(pq, (dist[start_r][start_c][0], start_r, start_c, 0))

        # Cost to reach (start_r, start_c) and go to floor F[start_r][start_c] is |start_y - F[start_r][start_c]|
        dist[start_r][start_c][1] = abs(start_y - F[start_r][start_c])
        heapq.heappush(pq, (dist[start_r][start_c][1], start_r, start_c, 1))

        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        while pq:
            cost, r, c, type_rc = heapq.heappop(pq)

            # If we found a cheaper path already, ignore this one
            if dist[r][c][type_rc] != -1 and cost > dist[r][c][type_rc]:
                continue

            # We are effectively at building (r, c) at the reference floor Y_rc
            # The cost `cost` is the minimum stair cost to reach building (r,c) AND adjust to floor Y_rc
            Y_rc = 1 if type_rc == 0 else F[r][c]

            # Explore neighbors (nr, nc)
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]

                if 0 <= nr < H and 0 <= nc < W:
                    
                    # Valid walkway floor levels L are from 1 up to min(F[r][c], F[nr][nc])
                    min_walkway_floor = 1
                    max_walkway_floor = min(F[r][c], F[nr][nc])

                    # Possible target states in neighbor (nr, nc) are reaching floor 1 (type 0) or floor F[nr][nc] (type 1)
                    for type_nrc in range(2):
                        Y_nrc = 1 if type_nrc == 0 else F[nr][nc]

                        # We are minimizing cost + |L - Y_rc| + |L - Y_nrc| over L in [min_walkway_floor, max_walkway_floor]
                        # The function f(L) = |L - Y_rc| + |L - Y_nrc| is convex.
                        # The minimum of f(L) over an interval [a, b] occurs at a, b, or critical points.
                        # Critical points of |L-X| + |L-Y| are X and Y.
                        # So, relevant L values to check are {min_walkway_floor, max_walkway_floor, Y_rc, Y_nrc},
                        # restricted to the interval [min_walkway_floor, max_walkway_floor].

                        candidate_L_values = set()
                        candidate_L_values.add(min_walkway_floor)
                        candidate_L_values.add(max_walkway_floor)
                        candidate_L_values.add(Y_rc)
                        candidate_L_values.add(Y_nrc)

                        min_additional_cost = float('inf')

                        for L in candidate_L_values:
                            if min_walkway_floor <= L <= max_walkway_floor:
                                # cost to reach floor L in (r,c) from Y_rc reference: |L - Y_rc|
                                # cost from floor L to Y_nrc reference floor in (nr,nc): |L - Y_nrc|
                                # Total additional stair cost for this transition via L: |L - Y_rc| + |L - Y_nrc|
                                min_additional_cost = min(min_additional_cost, abs(L - Y_rc) + abs(L - Y_nrc))
                        
                        # Total cost to reach (nr, nc) and go to floor Y_nrc via this path
                        total_cost_via_neighbor = cost + min_additional_cost

                        if total_cost_via_neighbor != float('inf'):
                            if dist[nr][nc][type_nrc] == -1 or total_cost_via_neighbor < dist[nr][nc][type_nrc]:
                                dist[nr][nc][type_nrc] = total_cost_via_neighbor
                                heapq.heappush(pq, (dist[nr][nc][type_nrc], nr, nc, type_nrc))

        # After Dijkstra, calculate the minimum cost to reach (end_r, end_c) at floor end_z
        ans = float('inf')

        # Option 1: reached (end_r, end_c) and went to floor 1 (cost dist[end_r][end_c][0]), then stairs from 1 to end_z
        if dist[end_r][end_c][0] != -1:
             ans = min(ans, dist[end_r][end_c][0] + abs(end_z - 1))

        # Option 2: reached (end_r, end_c) and went to floor F[end_r][end_c] (cost dist[end_r][end_c][1]), then stairs from F[end_r][end_c] to end_z
        if dist[end_r][end_c][1] != -1:
            ans = min(ans, dist[end_r][end_c][1] + abs(end_z - F[end_r][end_c]))

        return ans

    for query in queries:
        ans = min_stairs(*query)
        print(ans)

solve()