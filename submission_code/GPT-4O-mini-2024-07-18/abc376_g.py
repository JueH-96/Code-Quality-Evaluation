def modinv(a, p):
    """ Return the modular inverse of a under modulo p using Fermat's little theorem. """
    return pow(a, p - 2, p)

def expected_operations(N, parents, weights):
    MOD = 998244353
    
    # Create the tree structure
    tree = [[] for _ in range(N + 1)]
    for i in range(N):
        tree[parents[i]].append(i + 1)
    
    # Total weight
    total_weight = sum(weights)
    
    # Expected operations
    expected = [0] * (N + 1)
    
    # DFS to calculate expected operations
    def dfs(node):
        nonlocal total_weight
        sum_expected = 0
        child_count = 0
        
        for child in tree[node]:
            dfs(child)
            sum_expected += expected[child]
            child_count += 1
        
        if child_count == 0:
            expected[node] = 0
            return
        
        # Calculate expected value for this node
        weight = weights[node - 1] if node > 0 else 0
        expected[node] = (sum_expected + child_count) * weight * modinv(total_weight, MOD) % MOD
    
    dfs(0)
    
    # The expected number of operations is stored in expected[0]
    return expected[0]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        parents = list(map(int, data[index:index + N]))
        index += N
        weights = list(map(int, data[index:index + N]))
        index += N
        
        result = expected_operations(N, parents, weights)
        results.append(result)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()