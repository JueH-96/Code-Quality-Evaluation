# YOUR CODE HERE
import sys

# Increase recursion limit for deep DSU paths and get_next chains.
# Max N, Q are 2*10^5, so N+Q can be up to 4*10^5.
sys.setrecursionlimit(4 * 10**5 + 10)

def main():
    """
    Main function to solve the problem.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        N, Q = map(int, readline().split())
        
        ops = []
        for i in range(Q):
            L, R, C = map(int, readline().split())
            ops.append((C, L, R, i + 1))
            
    except (IOError, ValueError):
        # Handle potential empty input or format errors, though not expected by problem spec
        N, Q, ops = 0, 0, []

    if N + Q == 0:
        print(0)
        return

    # Sort operations by cost in ascending order (Kruskal's algorithm idea)
    ops.sort()
    
    total_vertices = N + Q
    
    # DSU (Disjoint Set Union) data structure for total_vertices (1-indexed)
    parent = list(range(total_vertices + 1))
    size = [1] * (total_vertices + 1)
    
    def find(i):
        # Path compression
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            # Union by size
            if size[root_i] < size[root_j]:
                root_i, root_j = root_j, root_i
            parent[root_j] = root_i
            size[root_i] += size[root_j]
            return True
        return False

    # This data structure helps to find the next vertex in a range [L, R]
    # that we need to consider. It allows skipping over contiguous blocks of 
    # already-connected vertices within the 1..N range.
    # next_v[i] will point to the smallest vertex index >= i that we might need to process.
    next_v = list(range(N + 2))
    
    def get_next(i):
        # Path compression for the next_v structure
        if next_v[i] == i:
            return i
        next_v[i] = get_next(next_v[i])
        return next_v[i]

    mst_cost = 0
    edges_count = 0
    
    for C, L, R, op_idx in ops:
        v_op = N + op_idx
        
        # In this operation, we have edges of cost C connecting v_op to all j in [L, R].
        # This is equivalent to being able to connect any pair of vertices from the set
        # {v_op} U {j | L <= j <= R} with an edge of cost C.
        # We add edges greedily (Kruskal's) to connect components.
        
        # First, connect all components that have vertices in the range [L, R].
        # We can do this by picking a pivot (e.g., L) and connecting all other
        # components in the range to it.
        # We iterate from L to R using get_next to efficiently skip over nodes
        # that are already in the same component as their predecessors.
        
        u = get_next(L)
        while u <= R:
            # Connect the component of u to the component of L.
            # If they are already connected, union() will do nothing and return False.
            if union(L, u):
                mst_cost += C
                edges_count += 1
            
            # After processing u, any future query can skip it.
            # We update next_v[u] to point to the next node to check after u.
            # This effectively "removes" u from the set of nodes to check.
            old_u = u
            u = get_next(u + 1)
            next_v[old_u] = u
            
        # Second, connect the operation vertex v_op to the (now unified) component(s)
        # of the range [L, R]. We just need to connect it to one vertex from the range,
        # for example, L.
        if union(L, v_op):
            mst_cost += C
            edges_count += 1
            
    # A connected graph with V vertices has a spanning tree with V-1 edges.
    # If we couldn't form a spanning tree, the graph is disconnected.
    # The number of edges in our MST is `edges_count`.
    if edges_count == total_vertices - 1:
        print(mst_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()