import sys

# Set a higher recursion limit for deep DFS traversals.
# Maximum N is 10^5.
sys.setrecursionlimit(2 * 10**5) 

def solve():
    N = int(sys.stdin.readline())

    adj = [[] for _ in range(N + 1)]
    # Read N-1 edges. If N=1, this loop doesn't run.
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Read C_i values. Store them in a 1-indexed array.
    # C_arr[0] will be unused (dummy value).
    C_arr_str = sys.stdin.readline().split()
    C_arr = [0] * (N + 1) 
    for i in range(N):
        C_arr[i+1] = int(C_arr_str[i])

    if N == 1:
        print(0)
        return

    # Arrays for DFS passes
    depth = [0] * (N + 1)             # depth[u]: distance from root (node 1) to u
    subtree_C_sum = [0] * (N + 1)     # subtree_C_sum[u]: sum of C_i in subtree rooted at u

    # DFS1: Calculate depths and subtree_C_sums.
    # u: current node, p: parent of u in DFS tree, d: current depth of u
    def dfs1(u, p, d):
        depth[u] = d
        current_sCs = C_arr[u] # Start with C_u for the sum
        for v_node in adj[u]:
            if v_node == p: # Don't go back to parent
                continue
            dfs1(v_node, u, d + 1)
            current_sCs += subtree_C_sum[v_node] # Add sum from child's subtree
        subtree_C_sum[u] = current_sCs

    # Start DFS1 from node 1 (arbitrary root).
    # Parent of root is 0 (dummy), depth of root is 0.
    dfs1(1, 0, 0)

    # TotalSumC is the sum of all C_i values in the tree.
    # This is equal to subtree_C_sum[1] (sum for the root of the DFS tree).
    total_C_sum_val = subtree_C_sum[1]
    
    # f_values[x] will store the calculated f(x) value.
    f_values = [0] * (N + 1) 
    
    # Calculate f_values[1] (i.e., f(root))
    f_root_val = 0
    for i in range(1, N + 1): # Iterate through all nodes
        f_root_val += C_arr[i] * depth[i]
    f_values[1] = f_root_val

    # DFS2: Calculate f_values for all other nodes using the recurrence.
    # u: current node, p: parent of u in DFS tree
    def dfs2(u, p):
        for v_node in adj[u]:
            if v_node == p: # Don't go back to parent
                continue
            
            # Recurrence relation: f(v) = f(u) + TotalSumC - 2 * W(S_v)
            # Here, v_node is a child of u in the DFS tree.
            # W(S_{v_node}) is subtree_C_sum[v_node].
            f_values[v_node] = f_values[u] + total_C_sum_val - 2 * subtree_C_sum[v_node]
            dfs2(v_node, u)

    # Start DFS2 from node 1 (root).
    dfs2(1, 0)

    # Find the minimum value in f_values.
    # f_values[0] is unused. We need min from f_values[1]...f_values[N].
    min_overall_f = f_values[1] # Initialize with f(root)
    for i in range(2, N + 1):
        if f_values[i] < min_overall_f:
            min_overall_f = f_values[i]
    
    print(min_overall_f)

solve()