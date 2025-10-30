import sys

# Increase recursion limit for deep DFS traversals
# Max N is 2*10^5. DFS depth can be N. timer_arr_ref[0] goes up to 2N for tin/tout.
# Python's default limit (e.g., 1000 or 3000) is too low.
sys.setrecursionlimit(4 * 10**5 + 500) 

# Global constant for logarithm of N, for binary lifting table
# Max N is 2*10^5. 2^17 = 131072, 2^18 = 262144.
# So, LOGN_CONST = 18 means table indices 0 to 17, covering jumps up to 2^17.
LOGN_CONST = 18

# DFS to compute depths, parent pointers (up[u][0]), and tin/tout times
def dfs(u, p, current_depth, adj_list, depth_arr, up_arr, tin_arr, tout_arr, timer_arr_ref):
    # timer_arr_ref is a list [value] to pass int by mutable reference
    timer_arr_ref[0] += 1
    tin_arr[u] = timer_arr_ref[0]
    
    depth_arr[u] = current_depth
    up_arr[u][0] = p # p is parent of u

    # Precompute up_arr[u][i] for i > 0 (binary lifting table)
    for i in range(1, LOGN_CONST):
        up_arr[u][i] = up_arr[up_arr[u][i-1]][i-1]

    for v_node in adj_list[u]:
        if v_node == p: # Don't go back to parent
            continue
        dfs(v_node, u, current_depth + 1, adj_list, depth_arr, up_arr, tin_arr, tout_arr, timer_arr_ref)
    
    timer_arr_ref[0] += 1
    tout_arr[u] = timer_arr_ref[0]

# Function to check if u is an ancestor of v using tin/tout times
def is_ancestor(u, v, tin_arr, tout_arr):
    return tin_arr[u] <= tin_arr[v] and tout_arr[u] >= tout_arr[v]

# Function to find Lowest Common Ancestor (LCA) of u and v
def lca(u, v, up_arr, tin_arr, tout_arr):
    if is_ancestor(u, v, tin_arr, tout_arr):
        return u
    if is_ancestor(v, u, tin_arr, tout_arr):
        return v
    
    for i in range(LOGN_CONST - 1, -1, -1):
        # up_arr[u][i] is the 2^i-th ancestor of u.
        # If this 2^i-th ancestor is NOT an ancestor of v, then we can safely jump u.
        if not is_ancestor(up_arr[u][i], v, tin_arr, tout_arr):
            u = up_arr[u][i]
            
    return up_arr[u][0]

def main():
    N, K_val = map(int, sys.stdin.readline().split())

    adj_list = [[] for _ in range(N + 1)]
    for _ in range(N - 1): # N-1 edges for a tree
        u_edge, v_edge = map(int, sys.stdin.readline().split())
        adj_list[u_edge].append(v_edge)
        adj_list[v_edge].append(u_edge)

    V_special_nodes = list(map(int, sys.stdin.readline().split()))

    if K_val == 1:
        print(1)
        return

    timer_arr_ref = [0] 
    depth_arr = [0] * (N + 1)
    up_arr = [[0] * LOGN_CONST for _ in range(N + 1)] 
    tin_arr = [0] * (N + 1)
    tout_arr = [0] * (N + 1)

    # Perform DFS starting from node 1 (arbitrary root).
    # Parent of root (node 1) is set to itself (node 1). Depth of root is 0.
    dfs(1, 1, 0, adj_list, depth_arr, up_arr, tin_arr, tout_arr, timer_arr_ref)
    
    V_special_nodes.sort(key=lambda v_node: tin_arr[v_node])

    sum_depth_ui = 0
    for v_node in V_special_nodes:
        sum_depth_ui += depth_arr[v_node]

    sum_depth_lca = 0
    for i in range(K_val):
        u_curr = V_special_nodes[i]
        u_next = V_special_nodes[(i + 1) % K_val] 
        
        common_ancestor = lca(u_curr, u_next, up_arr, tin_arr, tout_arr)
        sum_depth_lca += depth_arr[common_ancestor]
        
    ans = sum_depth_ui - sum_depth_lca + 1
    print(ans)

if __name__ == '__main__':
    main()