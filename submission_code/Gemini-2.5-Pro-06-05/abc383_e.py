import sys

# It is good practice to set a higher recursion limit for problems
# that might involve deep recursion, such as DSU on a large number of items.
sys.setrecursionlimit(2 * 10**5 + 5)

def main():
    """
    Main logic to solve the problem.
    """
    # Fast I/O
    input = sys.stdin.readline
    
    # Read problem parameters
    N, M, K = map(int, input().split())
        
    # Read edges
    edges = []
    for _ in range(M):
        u, v, w = map(int, input().split())
        edges.append((w, u, v))
    
    # Read sequences A and B
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Sort edges by weight in ascending order
    edges.sort()
    
    # Disjoint Set Union (DSU) data structure initialization
    parent = list(range(N + 1))
    # For union-by-size optimization
    sz = [1] * (N + 1)
    # Store counts of A and B sequence members in each component
    count_A = [0] * (N + 1)
    count_B = [0] * (N + 1)
    
    # Initialize counts for each vertex based on sequences A and B
    for node in A:
        count_A[node] += 1
    for node in B:
        count_B[node] += 1

    def find(i):
        # Path compression optimization
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    total_sum = 0
    
    # Process edges in increasing order of weight (like Kruskal's algorithm)
    for w, u, v in edges:
        root_u = find(u)
        root_v = find(v)
        
        if root_u != root_v:
            # This edge connects two different components, so it's in the MST.
            
            # Union-by-size optimization: always merge smaller component into larger
            if sz[root_u] < sz[root_v]:
                root_u, root_v = root_v, root_u

            # Get A/B counts for each component before merging
            c_A_u, c_B_u = count_A[root_u], count_B[root_u]
            c_A_v, c_B_v = count_A[root_v], count_B[root_v]
            
            # The number of pairs that could be formed within each component before the merge.
            # The cost for these pairs is determined by smaller weight edges.
            pairs_possible_in_u = min(c_A_u, c_B_u)
            pairs_possible_in_v = min(c_A_v, c_B_v)
            
            # Merge the components (root_v into root_u)
            parent[root_v] = root_u
            sz[root_u] += sz[root_v]
            count_A[root_u] += c_A_v
            count_B[root_u] += c_B_v

            # A/B counts for the new merged component
            c_A_new, c_B_new = count_A[root_u], count_B[root_u]

            # The total number of pairs possible in the new merged component
            pairs_possible_in_new = min(c_A_new, c_B_new)
            
            # The number of *newly* formable pairs due to this merge.
            # These are the pairs for which f(A_i, B_j) = w, as this edge is the bottleneck.
            new_pairs = pairs_possible_in_new - pairs_possible_in_u - pairs_possible_in_v
            
            total_sum += new_pairs * w

    print(total_sum)

if __name__ == "__main__":
    main()