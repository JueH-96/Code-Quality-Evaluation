import sys

def find(parent, i):
    # Iterative find with path compression
    root = i
    while parent[root] != root:
        root = parent[root]
    
    # Path compression
    curr = i
    while parent[curr] != root:
        next_node = parent[curr] # Store next node before changing parent
        parent[curr] = root
        curr = next_node
        
    return root

def union(parent, size, top_k_list, u, v):
    # Convert 1-based vertex numbers to 0-based indices
    root_u = find(parent, u - 1)
    root_v = find(parent, v - 1)

    if root_u != root_v:
        # Union by size: attach smaller tree to larger tree
        if size[root_u] < size[root_v]:
            root_u, root_v = root_v, root_u # Swap so root_u is the larger tree

        parent[root_v] = root_u
        size[root_u] += size[root_v]

        # Merge top_k_lists
        # Combine lists from both roots
        combined_list = top_k_list[root_u] + top_k_list[root_v]
        
        # Sort descending and take top 10 unique elements
        # Using a set first ensures uniqueness
        # Sorting and slicing takes care of getting the largest
        # The problem constraint k <= 10 allows us to keep just the top 10
        
        unique_set = set(combined_list)
        unique_list = list(unique_set)
        unique_list.sort(reverse=True)
        
        # Keep only the top M=10 elements. Slice handles lists shorter than 10.
        top_k_list[root_u] = unique_list[:10]
        
        # No need to clear top_k_list[root_v] explicitly as it's no longer a root

def solve():
    # Faster input reading
    input = sys.stdin.readline
    
    N, Q = map(int, input().split())

    parent = list(range(N))
    size = [1] * N
    # Each vertex starts in its own component, the list contains just the vertex number (1-based)
    top_k_list = [[i + 1] for i in range(N)] 

    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            # Type 1: 1 u v
            u, v = query[1], query[2]
            union(parent, size, top_k_list, u, v)
        else:
            # Type 2: 2 v k
            v, k = query[1], query[2]
            
            root_v = find(parent, v - 1)
            component_top_k = top_k_list[root_v]
            
            # Check if the list has at least k elements
            if len(component_top_k) < k:
                print(-1)
            else:
                # The list is already sorted descending, k-th largest is at index k-1
                print(component_top_k[k - 1])

solve()