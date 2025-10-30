def mod_inverse(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m

def solve_case():
    N = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    # Build tree adjacency list
    children = [[] for _ in range(N + 1)]
    for i in range(N):
        children[p[i]].append(i + 1)
    
    # Calculate total probability
    total = sum(a)
    
    # dp[v] = (expected operations, probability sum) for subtree rooted at v
    dp = [(0, 0) for _ in range(N + 1)]
    
    def dfs(v):
        prob_sum = 0
        if v != 0:
            prob_sum = a[v-1]
        
        if not children[v]:
            dp[v] = (0, prob_sum)
            return
        
        # Sort children by expected operations / probability
        child_ratios = []
        for child in children[v]:
            exp_ops, prob = dfs(child)
            if prob > 0:
                child_ratios.append((exp_ops / prob, child))
        
        child_ratios.sort()
        
        # Calculate expected operations
        curr_prob = prob_sum
        total_exp = 0
        
        for _, child in child_ratios:
            child_exp, child_prob = dp[child]
            total_exp += curr_prob * 1  # cost to search this vertex
            total_exp += child_exp
            curr_prob += child_prob
        
        dp[v] = (total_exp, curr_prob)
        return dp[v]
    
    exp_ops, _ = dfs(0)
    
    # Convert to modulo
    MOD = 998244353
    numerator = int(exp_ops)
    denominator = total
    
    # Calculate result modulo
    result = (numerator * mod_inverse(denominator, MOD)) % MOD
    return result

def main():
    T = int(input())
    for _ in range(T):
        print(solve_case())

main()