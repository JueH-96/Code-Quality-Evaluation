def main():
    import sys
    from itertools import product

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2 + N]))
    
    MOD = 998244353
    
    # Count the number of -1s in B
    q = B.count(-1)
    
    # Precompute the number of connected components for each possible configuration
    def count_components(sequence):
        # Create a graph representation
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if sequence[i] <= sequence[j]:
                    edges.append((i, j))
        
        # Use union-find to count connected components
        parent = list(range(N))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX
        
        for u, v in edges:
            union(u, v)
        
        # Count unique roots
        unique_roots = len(set(find(i) for i in range(N)))
        return unique_roots
    
    total_sum = 0
    
    # Generate all combinations for the -1 positions
    for replacements in product(range(1, M + 1), repeat=q):
        current_sequence = B[:]
        replacement_index = 0
        
        for i in range(N):
            if current_sequence[i] == -1:
                current_sequence[i] = replacements[replacement_index]
                replacement_index += 1
        
        total_sum += count_components(current_sequence)
        total_sum %= MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()