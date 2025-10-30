import sys

# It is necessary to increase the recursion limit for deep trees in Python.
# N can be up to 10^5, so the recursion depth can be N.
sys.setrecursionlimit(2 * 10**5)

def solve():
    """
    Reads the tree structure and costs, then computes the minimum f(v)
    using a two-pass DFS (rerooting technique).
    """
    try:
        # Use sys.stdin.readline for faster I/O.
        input = sys.stdin.readline
        
        N_str = input()
        # Handle potential empty input from some environments.
        if not N_str:
            return
        N = int(N_str)
        
        # Build an adjacency list for the tree.
        # The input vertices are 1-indexed, so we convert to 0-indexed.
        adj = [[] for _ in range(N)]
        for _ in range(N - 1):
            a, b = map(int, input().split())
            adj[a - 1].append(b - 1)
            adj[b - 1].append(a - 1)
            
        # Read the costs C and store them in a 0-indexed list.
        C = list(map(int, input().split()))
        
        # subtree_C_sum[u]: stores the sum of C_i for all i in the subtree of u.
        # The tree is arbitrarily rooted at node 0 for the first pass.
        subtree_C_sum = [0] * N
        
        # f_values[u]: stores the value of the function f(u).
        f_values = [0] * N
        
        # --- Pass 1: Post-order DFS from root 0 ---
        # This pass computes two things:
        # 1. subtree_C_sum[u] for every node u.
        # 2. f_values[0], the value of f for our chosen root.
        
        def dfs1(u, p, depth):
            """
            First DFS (post-order traversal logic).
            Computes subtree sums of C and the initial f(0).
            - u: current node (0-indexed)
            - p: parent node (0-indexed)
            - depth: depth of the current node from the root (0)
            """
            # Add the contribution of this node to f(0).
            f_values[0] += C[u] * depth
            
            # Initialize the subtree sum with its own C value.
            subtree_C_sum[u] = C[u]
            
            for v in adj[u]:
                if v == p:
                    continue
                # Recurse on children.
                dfs1(v, u, depth + 1)
                # After visiting a child's subtree, update the parent's sum.
                subtree_C_sum[u] += subtree_C_sum[v]

        # Start the first DFS from root 0. The parent of the root is indicated by -1.
        dfs1(0, -1, 0)
        
        # The total sum of all C values is the sum of the whole tree,
        # which is now stored in subtree_C_sum[0].
        S_total = subtree_C_sum[0]
        
        # --- Pass 2: Pre-order DFS from root 0 ---
        # This pass uses the results from Pass 1 to compute f(v) for all other nodes
        # using the rerooting formula.
        
        def dfs2(u, p):
            """
            Second DFS (pre-order traversal logic).
            Computes f(v) for all v using the rerooting formula.
            - u: current node (0-indexed)
            - p: parent node (0-indexed)
            """
            for v in adj[u]:
                if v == p:
                    continue
                # Apply the rerooting formula to calculate f(v) from f(u):
                # f(v) = f(u) + S_total - 2 * subtree_C_sum[v]
                f_values[v] = f_values[u] + S_total - 2 * subtree_C_sum[v]
                # Recurse to children to propagate the calculation.
                dfs2(v, u)

        # Start the second DFS from the same root 0.
        dfs2(0, -1)
        
        # The f_values array is now fully populated. Find and print the minimum.
        print(min(f_values))

    except (IOError, ValueError):
        # Graceful exit for empty input or malformed lines.
        return

# Call the main function to run the solution.
solve()