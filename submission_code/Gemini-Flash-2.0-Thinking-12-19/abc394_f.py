import sys
from collections import defaultdict

sys.setrecursionlimit(300000)

def solve():
    N = int(sys.stdin.readline())
    adj = defaultdict(list)
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # DP states:
    # dp[u][state][has_deg4]
    # state:
    # 0: u not included. Valid alkane subgraph entirely within u's subtree.
    # 1: u included, connected to parent, requires 0 children connections (total deg 1)
    # 2: u included, connected to parent, requires 3 children connections (total deg 4)
    # 3: u included, not connected to parent, requires 1 child connection (total deg 1)
    # 4: u included, not connected to parent, requires 4 children connections (total deg 4)
    # has_deg4: 0 if no degree 4 vertex found yet, 1 if at least one degree 4 vertex found.

    # Initialize DP table with -1 (impossible)
    # dp[u][state][has_deg4]
    dp = [[[-1 for _ in range(2)] for _ in range(5)] for _ in range(N + 1)]

    # dp[u][0][0] = 0 means an empty subgraph within u's subtree is possible, size 0, no deg4.
    for i in range(1, N + 1):
        dp[i][0][0] = 0

    def dfs(u, parent):
        # Compute DP for children first
        children = [v for v in adj[u] if v != parent]
        for v in children:
            dfs(v, u)

        # --- Compute dp[u][0][has_deg4] ---
        # An alkane subgraph in u's subtree not including u explicitly connected upwards
        # must be entirely contained within one of its children's subtrees.
        # Max vertices from any valid alkane in a child's subtree.
        for v in children:
            for state in range(5):
                for has_deg4 in range(2):
                    if dp[v][state][has_deg4] != -1:
                        dp[u][0][has_deg4] = max(dp[u][0][has_deg4], dp[v][state][has_deg4])

        # --- Compute dp[u][1..4][has_deg4] using map DP ---
        # u is included. map_dp[conn_parent][has_deg4_so_far][down_degree_count] = vertices_from_children
        # conn_parent: 0 (not conn up), 1 (conn up)
        # has_deg4_so_far: 0 or 1
        # down_degree_count: 0 to 4 (max children connections needed for u's degree 1 or 4)
        map_dp = [[defaultdict(lambda: -1) for _ in range(2)] for _ in range(2)]

        # Base case for map_dp: u included, 0 children processed. 0 vertices from children.
        map_dp[0][0][0] = 0 # u not conn up, no deg4 yet, 0 children conn
        map_dp[1][0][0] = 0 # u conn up, no deg4 yet, 0 children conn

        for v in children:
            # Values from child v if connected to u (v must connect upwards to u)
            # v state 1: v conn up, deg 1, needs 0 down. dp[v][1][has_deg4]
            # v state 2: v conn up, deg 4, needs 3 down. dp[v][2][has_deg4] implies has_deg4=1

            v_gain = [-1, -1] # v_gain[has_deg4]

            # Child v connects up, result does not have deg4 yet (except potentially from u)
            if dp[v][1][0] != -1: v_gain[0] = dp[v][1][0]

            # Child v connects up, result has deg4 (either from v itself state 2, or from below state 1)
            if dp[v][1][1] != -1: v_gain[1] = max(v_gain[1], dp[v][1][1])
            if dp[v][2][1] != -1: v_gain[1] = max(v_gain[1], dp[v][2][1])

            next_map_dp = [[defaultdict(lambda: -1) for _ in range(2)] for _ in range(2)]

            for cp in range(2):
                for d4_prev in range(2):
                    for d_prev in range(5): # Iterate through possible previous down degrees 0..4
                        if map_dp[cp][d4_prev].get(d_prev, -1) == -1: continue

                        val = map_dp[cp][d4_prev][d_prev]

                        # Option 1: Don't connect to child v
                        next_map_dp[cp][d4_prev][d_prev] = max(next_map_dp[cp][d4_prev][d_prev], val)

                        # Option 2: Connect to child v
                        if d_prev + 1 <= 4: # Max 4 children connections needed for u's degree
                            if v_gain[0] != -1: # Child subtree has no deg4
                                next_map_dp[cp][d4_prev][d_prev + 1] = max(next_map_dp[cp][d4_prev][d_prev + 1], val + v_gain[0])
                            if v_gain[1] != -1: # Child subtree has deg4
                                next_map_dp[cp][d4_prev | 1][d_prev + 1] = max(next_map_dp[cp][d4_prev | 1][d_prev + 1], val + v_gain[1])

            map_dp = next_map_dp

        # After processing all children, populate dp[u][1..4]
        # dp[u][state][has_deg4] = map_dp[conn_parent][has_deg4_so_far][required_down_degree] + 1 (for u itself)

        # State 1: u conn up, total deg 1, needs 0 down. u is deg 1, does not provide deg4 itself.
        if map_dp[1][0].get(0, -1) != -1: dp[u][1][0] = map_dp[1][0][0] + 1
        if map_dp[1][1].get(0, -1) != -1: dp[u][1][1] = map_dp[1][1][0] + 1

        # State 2: u conn up, total deg 4, needs 3 down. u itself is deg 4 here, provides deg4.
        if map_dp[1][0].get(3, -1) != -1: dp[u][2][1] = max(dp[u][2][1], map_dp[1][0][3] + 1)
        if map_dp[1][1].get(3, -1) != -1: dp[u][2][1] = max(dp[u][2][1], map_dp[1][1][3] + 1)

        # State 3: u not conn up, total deg 1, needs 1 down. u is deg 1, does not provide deg4 itself.
        if map_dp[0][0].get(1, -1) != -1: dp[u][3][0] = map_dp[0][0][1] + 1
        if map_dp[0][1].get(1, -1) != -1: dp[u][3][1] = map_dp[0][1][1] + 1

        # State 4: u not conn up, total deg 4, needs 4 down. u itself is deg 4 here, provides deg4.
        if map_dp[0][0].get(4, -1) != -1: dp[u][4][1] = max(dp[u][4][1], map_dp[0][0][4] + 1)
        if map_dp[0][1].get(4, -1) != -1: dp[u][4][1] = max(dp[u][4][1], map_dp[0][1][4] + 1)


    # Start DFS from root 1
    dfs(1, 0)

    # The maximum number of vertices in a valid alkane is the maximum over all nodes u
    # of states that represent a self-contained valid alkane (has_deg4=1).
    # These are states 0, 3, and 4.
    # State 0: Valid alkane within subtree, not involving u connecting upwards.
    # State 3: Valid alkane rooted at u, u deg 1, not conn up, has deg4 somewhere below.
    # State 4: Valid alkane rooted at u, u deg 4, not conn up, has deg4 (u itself or from below).

    max_vertices = -1

    for u in range(1, N + 1):
        # Consider state 0: u not included, valid alkane in subtree
        if dp[u][0][1] != -1:
            max_vertices = max(max_vertices, dp[u][0][1])

        # Consider state 3: u is root of subgraph, u deg 1, not conn up, has deg4 somewhere below
        if dp[u][3][1] != -1:
             max_vertices = max(max_vertices, dp[u][3][1])

        # Consider state 4: u is root of subgraph, u deg 4, not conn up, has deg4 (u itself)
        if dp[u][4][1] != -1:
             max_vertices = max(max_vertices, dp[u][4][1])

    print(max_vertices)

solve()