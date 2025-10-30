import sys

sys.setrecursionlimit(300000)

def get_target_r2(N):
    if N < 5: return -1 # Alkane requires N >= 5
    if N % 3 != 2: return -1 # N-2 must be divisible by 3
    k = (N - 2) // 3
    return k % 3

# Map key: (n%9, n4%3, has_n4)
# Map value: max_vertices

def merge_maps(map1, map2):
    if not map1: return dict(map2)
    if not map2: return dict(map1)

    merged = {}
    for (r1_1, r2_1, h1), n1 in map1.items():
        for (r1_2, r2_2, h2), n2 in map2.items():
            new_r1 = (r1_1 + r1_2) % 9
            new_r2 = (r2_1 + r2_2) % 3
            new_h = h1 or h2
            key = (new_r1, new_r2, new_h)
            merged[key] = max(merged.get(key, -1), n1 + n2)
    return merged

# DP state indices:
# dp[u][0]: max size of self-contained alkane in subtree of u (integer)
# dp[u][1]: map for structure rooted at u, wants deg 1 connected up (p=1, T=1), deficit 0
# dp[u][2][d]: map for structure rooted at u, wants deg 4 connected up (p=1, T=4), deficit d (d=0..3)
# dp[u][3][d]: map for structure rooted at u, wants deg 1 no up conn (p=0, T=1), deficit d (d=0..1)
# dp[u][4][d]: map for structure rooted at u, wants deg 4 no up conn (p=0, T=4), deficit d (d=0..4)

adj = []
deg = []
dp = [] # Stores maps for states 1, 2, 3, 4
dp_alkane = [] # Stores int for state 0
N = 0

def dfs(u, parent):
    global dp, dp_alkane

    dp_alkane[u] = -1

    # tmp_dp[state_type][d] maps (n%9, n4%3, has_n4) -> max_vertices
    # state_type: 1->(p=1, T=1), 2->(p=1, T=4), 3->(p=0, T=1), 4->(p=0, T=4)
    # d: deficit
    # Max deficit: state 1 needs 0, state 2 needs 3, state 3 needs 1, state 4 needs 4.
    # Max d index is 4.
    tmp_dp = [[{} for _ in range(5)] for _ in range(5)] # tmp_dp[1..4][0..4]

    # Initialize tmp_dp for u alone (n=1)
    n0_rem9 = 1 % 9 # 1
    n0_rem3 = 1 % 3 # 1

    if deg[u] == 1: # u can only be deg 1 leaf in alkane, connected up (p=1, T=1) if included
        # needs 0 children, deficit 0
        # state_type 1, deficit 0
        tmp_dp[1][0][(n0_rem9, 0 % 3, False)] = 1 # n=1, n4=0, has_n4=0
    else: # deg[u] > 1, u can be deg 4 if included
        # u wants deg 4 up (p=1, T=4), needs 3 children, deficit 3
        # state_type 2, deficit 3
        tmp_dp[2][3][(n0_rem9, 1 % 3, True)] = 1 # n=1, n4=1, has_n4=1
        # u wants deg 4 no up (p=0, T=4), needs 4 children, deficit 4
        # state_type 4, deficit 4
        tmp_dp[4][4][(n0_rem9, 1 % 3, True)] = 1 # n=1, n4=1, has_n4=1

    children = [v for v in adj[u] if v != parent]

    for v in children:
        dfs(v, u)

        # Update self-contained alkane maximum from child v's subtree
        dp_alkane[u] = max(dp_alkane[u], dp_alkane[v])

        next_tmp_dp = [[{} for _ in range(5)] for _ in range(5)]

        # Option 1: v's subtree doesn't connect to u.
        # The states for u regarding its children from v_1..v_i-1 remain the same.
        # Just copy tmp_dp to next_tmp_dp.
        for state_type in range(1, 5):
            # Determine max deficit for this state_type
            max_d = {1:0, 2:3, 3:1, 4:4}.get(state_type, 0)
            for d in range(max_d + 1):
                 next_tmp_dp[state_type][d] = dict(tmp_dp[state_type][d])

        # Option 2: v connects to u.
        # v must provide a structure needing 1 connection upwards.
        # These structures are in dp[v][1][0] (v deg 1 up, deficit 0)
        # and dp[v][2][0] (v deg 4 up, deficit 0).
        M_v_deg1_up = dp[v][1][0] # Map from (r1', r2', h') -> N'
        M_v_deg4_up = dp[v][2][0] # Map from (r1', r2', h') -> N'

        # Iterate through all current states (state_type, d) with map M_u = tmp_dp[state_type][d]
        for state_type in range(1, 5):
            # Determine max deficit for this state_type
            max_d = {1:0, 2:3, 3:1, 4:4}.get(state_type, 0)
            for d in range(max_d + 1):
                 M_u = tmp_dp[state_type][d]
                 if not M_u: continue # Skip if current state is empty

                 # If deficit > 0, v can contribute to reduce deficit
                 if d > 0:
                     # Try combining with v's deg 1 up structure (M_v_deg1_up)
                     if M_v_deg1_up:
                         merged = merge_maps(M_u, M_v_deg1_up)
                         current_map = next_tmp_dp[state_type][d-1]
                         for key, val in merged.items():
                             current_map[key] = max(current_map.get(key, -1), val)
                         next_tmp_dp[state_type][d-1] = current_map

                     # Try combining with v's deg 4 up structure (M_v_deg4_up)
                     if M_v_deg4_up:
                         merged = merge_maps(M_u, M_v_deg4_up)
                         current_map = next_tmp_dp[state_type][d-1]
                         for key, val in merged.items():
                             current_map[key] = max(current_map.get(key, -1), val)
                         next_tmp_dp[state_type][d-1] = current_map

        tmp_dp = next_tmp_dp

    # After processing all children, finalize dp[u] from tmp_dp
    
    # dp[u][1][0]: (p=1, T=1, d=0). Needs 0 children, deficit 0.
    # Only possible if original deg[u] == 1.
    if deg[u] == 1:
        dp[u][1][0] = tmp_dp[1][0]
    else:
        dp[u][1][0] = {} # Impossible state

    # dp[u][2][d]: (p=1, T=4, d). Needs 3 children. Deficit 3..0.
    # Possible d values 0..3. tmp_dp state_type 2.
    for d in range(4): # d = 0, 1, 2, 3
         dp[u][2][d] = tmp_dp[2][d]

    # dp[u][3][d]: (p=0, T=1, d). Needs 1 child. Deficit 1..0.
    # Possible d values 0..1. tmp_dp state_type 3.
    for d in range(2): # d = 0, 1
         dp[u][3][d] = tmp_dp[3][d]

    # dp[u][4][d]: (p=0, T=4, d). Needs 4 children. Deficit 4..0.
    # Possible d values 0..4. tmp_dp state_type 4.
    for d in range(5): # d = 0, 1, 2, 3, 4
         dp[u][4][d] = tmp_dp[4][d]

    # Check self-contained alkanes rooted at u
    # These come from states with deficit 0 (all required children connected) and p=0

    # Case: u wants deg 1 no up conn (p=0, T=1), needs 1 child, deficit 0.
    # Corresponding tmp_dp state_type 3, deficit 0.
    for (r1, r2, h), N in tmp_dp[3][0].items():
         if N >= 5 and h and r1 % 3 == 2:
             target_r2 = get_target_r2(N)
             if target_r2 != -1 and r2 == target_r2:
                dp_alkane[u] = max(dp_alkane[u], N)

    # Case: u wants deg 4 no up conn (p=0, T=4), needs 4 children, deficit 0.
    # Corresponding tmp_dp state_type 4, deficit 0.
    for (r1, r2, h), N in tmp_dp[4][0].items():
         if N >= 5 and h and r1 % 3 == 2:
             target_r2 = get_target_r2(N)
             if target_r2 != -1 and r2 == target_r2:
                dp_alkane[u] = max(dp_alkane[u], N)


def solve():
    global N, adj, deg, dp, dp_alkane
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]
    deg = [0] * (N + 1)
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # Initialize DP tables
    # dp[u][state_type][d] is a map
    # state_type: 1->(p=1, T=1), 2->(p=1, T=4), 3->(p=0, T=1), 4->(p=0, T=4)
    # Max deficits: 1->0, 2->3, 3->1, 4->4. Max index needed for d is 4.
    # dp[u][1][0] needs map
    # dp[u][2][0..3] need maps
    # dp[u][3][0..1] need maps
    # dp[u][4][0..4] need maps
    # Total unique (state_type, d) pairs needing maps: 1 + 4 + 2 + 5 = 12.
    # The current structure `dp = [[{} for _ in range(5)] for _ in range(N + 1)]` provides dp[u][0..4][0..4], which is 25 maps per node. This is fine.

    dp = [[{} for _ in range(5)] for _ in range(N + 1)]
    dp_alkane = [-1] * (N + 1) # Stores max alkane size in subtree rooted at u, initialized to -1

    dfs(1, 0) # Root at 1

    # The max alkane size for the whole tree is the max value in dp_alkane over all nodes.
    # However, the DP definition ensures dp_alkane[u] considers all alkanes fully contained
    # within u's subtree. Thus, dp_alkane[1] should contain the maximum for the whole tree.
    print(dp_alkane[1])

solve()