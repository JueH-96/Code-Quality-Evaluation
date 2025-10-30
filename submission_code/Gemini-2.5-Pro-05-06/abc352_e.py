import sys

# Increase recursion limit (though default is likely sufficient for DSU with path compression)
sys.setrecursionlimit(10**6) # A sufficiently large value

# Use fast I/O
input = sys.stdin.readline

# DSU data structures (module-level)
# These will be initialized by DSU_init()
parent = []
sz = [] # sz[i] stores the size of the component if i is a root

def DSU_init(N_nodes):
    global parent, sz # Needed because we are re-assigning the global variables parent and sz
    parent = list(range(N_nodes + 1)) # 1-indexed: parent[0] unused, parent[1] through parent[N_nodes]
    sz = [1] * (N_nodes + 1)      # sz[0] unused, initially each component has size 1

def find(i):
    # Path compression: make each node on the path point directly to the root
    # Accesses global 'parent' list but only modifies its elements, not the 'parent' reference itself.
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    # Union by size: attach the smaller tree under the root of the larger tree
    # Accesses global 'parent' and 'sz' lists, modifies their elements.
    # Returns True if merge happened (i and j were in different components), False otherwise.
    
    root_i = find(i)
    root_j = find(j)
    
    if root_i != root_j:
        # Ensure root_i is the representative of the larger (or equal size) component
        if sz[root_i] < sz[root_j]:
            root_i, root_j = root_j, root_i 
        
        parent[root_j] = root_i # Make root_j's component a child of root_i's component
        sz[root_i] += sz[root_j] # Update size of the new root's component
        return True
    return False # Already in the same component

def solve():
    N, M = map(int, input().split())

    DSU_init(N) # Initialize DSU structures for N vertices

    events = []
    for _ in range(M):
        line = list(map(int, input().split()))
        # K_val = line[0] # Number of vertices in S_i, not directly used after S_val is formed
        C_val = line[1]   # Cost for this operation
        S_val = line[2:]  # List of vertices in S_i
        events.append((C_val, S_val))

    # Sort events by cost C_val in non-decreasing order
    events.sort()

    mst_total_weight = 0
    num_edges = 0

    for C_val, S_val in events:
        # Optimization: if N-1 edges are already added, MST is complete.
        # No need to process events with higher or equal costs.
        if num_edges == N - 1:
            break
            
        # S_val is guaranteed by constraints (K_i >= 2) to have at least two elements.
        # Pick the first vertex in S_val as a reference node for this event's connections.
        u_node = S_val[0]
        # Try to connect u_node with all other K_i-1 vertices in S_val.
        for k_idx in range(1, len(S_val)): # Iterate from the second vertex in S_val
            v_node = S_val[k_idx]
            
            # If u_node and v_node are in different components, union them.
            # The union operation returns True if a merge occurred.
            if union(u_node, v_node):
                mst_total_weight += C_val
                num_edges += 1
                
                # An optimization: if MST becomes complete mid-event processing,
                # we could break from this inner loop. The outer loop's break condition
                # `if num_edges == N - 1: break` will then immediately trigger.
                # For simplicity, allowing the inner loop to finish for the current S_val
                # is also correct as all edges from this S_val have the same cost C_val.
                # if num_edges == N - 1:
                #    break
    
    # After processing all relevant events (or breaking early if MST is complete):
    if num_edges == N - 1: # A spanning tree for N vertices has N-1 edges
        print(mst_total_weight)
    else: # Graph is not connected if fewer than N-1 edges were added
        print("-1")

# Call the solver function
solve()