MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m - 2, m)

def solve():
    n = int(input())
    parents = [0] + list(map(int, input().split()))
    a = [0] + list(map(int, input().split()))
    
    # Build adjacency list
    children = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        children[parents[i]].append(i)
    
    # Calculate subtree sums and sizes
    subtree_sum = [0] * (n + 1)
    subtree_size = [0] * (n + 1)
    
    def dfs1(v):
        subtree_sum[v] = a[v]
        subtree_size[v] = 1
        for child in children[v]:
            dfs1(child)
            subtree_sum[v] += subtree_sum[child]
            subtree_size[v] += subtree_size[child]
    
    dfs1(0)
    
    total_sum = subtree_sum[0]
    
    # Calculate expected operations
    def dfs2(v):
        if not children[v]:
            return 0
        
        # Sort children by density (sum/size) in descending order
        children[v].sort(key=lambda c: subtree_sum[c] * subtree_size[v] - subtree_sum[v] * subtree_size[c], reverse=True)
        
        expected = 0
        cumulative_ops = 0
        cumulative_prob = 0
        
        for child in children[v]:
            # Expected operations if treasure is in this child's subtree
            child_expected = dfs2(child)
            
            # Probability that treasure is in this child's subtree
            prob = subtree_sum[child]
            
            # Add contribution: 
            # - Operations to reach this child (cumulative_ops + 1)
            # - Expected operations within the child's subtree
            expected += prob * (cumulative_ops + 1 + child_expected)
            
            # Update cumulative values
            cumulative_ops += subtree_size[child]
            cumulative_prob += prob
        
        return expected
    
    # Start from children of root
    expected_ops = 0
    cumulative_ops = 0
    
    # Sort root's children by density
    children[0].sort(key=lambda c: subtree_sum[c] * subtree_size[0] - subtree_sum[0] * subtree_size[c], reverse=True)
    
    for child in children[0]:
        child_expected = dfs2(child)
        prob = subtree_sum[child]
        expected_ops += prob * (cumulative_ops + 1 + child_expected)
        cumulative_ops += subtree_size[child]
    
    # Convert to modular arithmetic
    # expected_ops / total_sum
    result = (expected_ops % MOD) * modinv(total_sum % MOD) % MOD
    return result

t = int(input())
for _ in range(t):
    print(solve())