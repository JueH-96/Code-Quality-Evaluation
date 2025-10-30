import sys

# Increase recursion limit for deep trees
sys.setrecursionlimit(2 * 10**5 + 500)

def solve():
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # global_max_alkane_size stores the maximum number of vertices
    # found in any valid alkane subgraph.
    # Initialized to -1, which is the output if no alkane is found.
    global_max_alkane_size = -1

    def dfs(u, p):
        nonlocal global_max_alkane_size

        # deg_counts[i] stores the maximum number of vertices in a partial alkane
        # involving node 'u' and some of its descendants (within its subtree),
        # such that 'u' connects to exactly 'i' of its children.
        # The value 'deg_counts[i]' includes 'u' itself.
        # Initialize with 'u' itself, connected to 0 children initially.
        deg_counts = [-float('inf')] * 5
        deg_counts[0] = 1 # 'u' itself, 0 children connected so far

        for v in adj[u]:
            if v == p:
                continue

            # Recursively call DFS for child 'v'
            # v_val_deg1_up: Max size if v connects to u (as its parent), forming degree 1.
            # v_val_deg4_up: Max size if v connects to u (as its parent), forming degree 4.
            v_val_deg1_up, v_val_deg4_up = dfs(v, u)

            # Create a new temporary DP array to store results after considering child 'v'
            new_deg_counts = [-float('inf')] * 5

            # Iterate through all possible previous states (number of children 'u' was connected to)
            for i in range(5):
                if deg_counts[i] == -float('inf'):
                    continue

                # Option 1: 'u' does NOT connect to child 'v'.
                # The alkane component containing 'u' doesn't grow via 'v'.
                # 'v' and its subtree might form a separate alkane (handled by recursive call to dfs(v,u) updating global_max_alkane_size).
                new_deg_counts[i] = max(new_deg_counts[i], deg_counts[i])

                # Option 2: 'u' DOES connect to child 'v'.
                # 'u' uses one more connection slot. So, new state is i+1.
                # 'v' must satisfy its degree constraint by connecting 'up' to 'u'.
                if i + 1 <= 4: # 'u' can take one more child connection (max 4 total children connections)
                    # If 'v' becomes a degree-1 node by connecting to 'u'
                    if v_val_deg1_up != -float('inf'):
                        new_deg_counts[i + 1] = max(new_deg_counts[i + 1], deg_counts[i] + v_val_deg1_up)
                    
                    # If 'v' becomes a degree-4 node by connecting to 'u'
                    if v_val_deg4_up != -float('inf'):
                        new_deg_counts[i + 1] = max(new_deg_counts[i + 1], deg_counts[i] + v_val_deg4_up)
            
            deg_counts = new_deg_counts # Update deg_counts for the next child

        # After processing all children of 'u', determine the return values for 'u'
        # and update the global max alkane size based on self-contained alkanes rooted at 'u'.

        # val_deg1_up_for_u: 'u' connects to its parent, and its total degree is 1.
        # This means 'u' must have connected to 0 children.
        val_deg1_up_for_u = -float('inf')
        if deg_counts[0] != -float('inf'):
            val_deg1_up_for_u = deg_counts[0]

        # val_deg4_up_for_u: 'u' connects to its parent, and its total degree is 4.
        # This means 'u' must have connected to 3 children.
        val_deg4_up_for_u = -float('inf')
        if deg_counts[3] != -float('inf'):
            val_deg4_up_for_u = deg_counts[3]

        # Update global_max_alkane_size for alkanes that are self-contained within 'u's subtree
        # and do NOT connect to 'u's parent.
        # Such an alkane's root ('u') must be degree 4 (to satisfy condition 3).
        
        # If 'u' is a degree-4 root (not connecting to parent), it must connect to 4 children.
        if deg_counts[4] != -float('inf'):
            global_max_alkane_size = max(global_max_alkane_size, deg_counts[4])
        
        # Note: A degree-1 root (deg_counts[1]) would not satisfy the "at least one vertex of degree 4" condition
        # unless it's part of a larger alkane where some descendant is deg 4.
        # However, this DP formulation already ensures that `deg_counts[4]` values only propagate when valid deg4 ancestors are available,
        # or it is itself a root. So, only explicitly checking `deg_counts[4]` for the global max is sufficient.

        return val_deg1_up_for_u, val_deg4_up_for_u

    # Start DFS from node 1 (arbitrary root) with a dummy parent 0.
    # The return values from this initial call are not directly used for the final answer.
    # The global_max_alkane_size variable accumulates the maximum.
    dfs(1, 0)

    print(global_max_alkane_size)

solve()