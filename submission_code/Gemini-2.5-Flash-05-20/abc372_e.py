import sys

def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())

    # Initialize Disjoint Set Union (DSU) structures
    # parent[i]: stores the parent of element i. If parent[i] == i, then i is a root.
    parent = list(range(N + 1))
    # size[i]: stores the size of the component if i is a root. Used for union by size optimization.
    size = [1] * (N + 1)
    
    # top_k_values[i]: stores a sorted list (descending) of the at most 10 largest vertex numbers
    # in the component rooted at i. Max k from problem is 10, so we only need to store up to 10.
    top_k_values = []
    for i in range(N + 1):
        # Each vertex initially forms its own component, containing only itself.
        # Vertex 0 is unused, its list can be empty or [0] as it won't be queried.
        top_k_values.append([i]) 

    # Find operation with path compression
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i]) # Path compression
        return parent[i]

    # Union operation with union by size and updating top_k_values
    def union(i, j):
        root_i = find(i)
        root_j = find(j)

        if root_i != root_j:
            # Union by size: attach the root of the smaller tree to the root of the larger tree
            if size[root_i] < size[root_j]:
                root_i, root_j = root_j, root_i # Swap to ensure root_i is the root of the larger component

            # Merge the top_k_values lists of the two components
            # 1. Concatenate lists
            # 2. Convert to set to remove duplicates (important if vertices are present in both top lists,
            #    e.g. they are common high-value elements in merged component despite distinct paths)
            # 3. Convert back to list
            # 4. Sort in descending order
            # 5. Slice to keep only the top 10 elements (since max k is 10, this is sufficient)
            merged_list = sorted(list(set(top_k_values[root_i] + top_k_values[root_j])), reverse=True)
            
            # The new root (root_i) gets the merged and truncated list
            top_k_values[root_i] = merged_list[:10] # Max elements to store is 10

            # Attach the smaller tree (root_j) to the larger tree (root_i)
            parent[root_j] = root_i
            # Update the size of the new larger component
            size[root_i] += size[root_j]
            # top_k_values[root_j] becomes stale, but will not be accessed as root_j is no longer a root.

    results = [] # To store answers for Type 2 queries

    # Process Q queries
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        query_type = query[0]

        if query_type == 1:
            u, v = query[1], query[2]
            union(u, v)
        else: # query_type == 2
            v, k = query[1], query[2]
            
            # Find the root of the component containing vertex v
            root_v = find(v)
            # Get the pre-maintained top_k_values for this component
            component_top_k = top_k_values[root_v]
            
            # Check if there are at least k elements in the list
            if len(component_top_k) >= k:
                # k-th largest element is at index k-1 (0-indexed)
                results.append(str(component_top_k[k - 1]))
            else:
                # Fewer than k elements, print -1
                results.append("-1")

    # Print all results, each on a new line
    sys.stdout.write("
".join(results) + "
")

# Call the main function to execute the solution
if __name__ == '__main__':
    solve()