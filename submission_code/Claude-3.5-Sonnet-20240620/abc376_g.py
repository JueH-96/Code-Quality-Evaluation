# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def solve_case():
    N = int(input())
    parents = list(map(int, input().split()))
    probabilities = list(map(int, input().split()))
    
    total_prob = sum(probabilities)
    inv_total_prob = mod_inverse(total_prob)
    
    children = [[] for _ in range(N + 1)]
    for i, p in enumerate(parents, 1):
        children[p].append(i)
    
    dp = [0] * (N + 1)
    subtree_prob = [0] * (N + 1)
    
    def dfs(node):
        node_prob = probabilities[node - 1] if node > 0 else 0
        subtree_prob[node] = node_prob
        
        for child in children[node]:
            dfs(child)
            child_prob = subtree_prob[child]
            subtree_prob[node] += child_prob
            dp[node] += dp[child] + child_prob
        
        if node > 0:
            dp[node] = (dp[node] * inv_total_prob + 1) * total_prob
            dp[node] %= MOD
    
    dfs(0)
    return dp[0]

T = int(input())
for _ in range(T):
    result = solve_case()
    print(result)