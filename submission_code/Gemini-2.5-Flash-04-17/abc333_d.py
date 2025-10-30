import sys
sys.setrecursionlimit(300005) # Adjust recursion depth for potentially deep trees

def get_component_size(u, excluded_node, adj, visited):
    """
    DFS to count nodes in the component containing u, avoiding excluded_node.
    visited array must be fresh for each new component traversal, and
    excluded_node must be marked visited before the initial call.
    """
    visited[u] = True
    count = 1 # Count the current node u
    for v in adj[u]:
        # Check if v is the excluded node or already visited in this traversal
        if v != excluded_node and not visited[v]:
            count += get_component_size(v, excluded_node, adj, visited)
    return count

def solve():
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)] # 1-indexed adjacency list
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # If vertex 1 is a leaf (degree 1), it can be deleted in 1 operation.
    if len(adj[1]) == 1:
        print(1)
        return

    neighbors_of_1 = adj[1]
    max_comp_size = 0

    # To delete vertex 1 (with degree > 1), we must remove all its neighbors
    # except one, along with the components they belong to (after removing edge to 1).
    # We choose to keep the branch (component) attached to the neighbor v that
    # has the largest size. The number of vertices to remove is (N - 1 - size_of_v's_component)
    # plus vertex 1 itself. Total operations = (N - 1 - size_of_v's_component) + 1
    # = N - size_of_v's_component. We want to minimize this, so maximize size_of_v's_component.

    # For each neighbor v of 1, calculate the size of the connected component
    # containing v after removing vertex 1 and its incident edges.
    for neighbor in neighbors_of_1:
        # Need a fresh visited array for each component calculation
        # Size N+1 for 1-indexed nodes
        visited = [False] * (N + 1)
        
        # Mark 1 as visited to prevent DFS from neighbor from crossing to 1.
        # This effectively simulates removing vertex 1.
        visited[1] = True 
        
        # Call DFS starting from neighbor, count nodes reachable without visiting 1
        comp_size = get_component_size(neighbor, 1, adj, visited)
        max_comp_size = max(max_comp_size, comp_size)

    # The minimum operations required is N - max_comp_size.
    # This is the total number of vertices minus the largest component size
    # attached to vertex 1 (which we keep), plus 1 for vertex 1 itself.
    # The vertices removed are N - (max_comp_size + 1) vertices
    # from other components, plus 1 for vertex 1.
    # This is (N - 1 - max_comp_size) + 1 = N - max_comp_size operations.
    print(N - max_comp_size)

# Read input and solve
solve()