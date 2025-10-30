def solve_case():
    n = int(input())
    if n == 0:
        return 0
    
    parents = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    # Build the tree
    children = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        children[parents[i - 1]].append(i)
    
    from fractions import Fraction
    
    # Calculate subtree weights (probability mass in each subtree)
    def calc_subtree_weight(v):
        if v == 0:
            weight = Fraction(0)
        else:
            weight = Fraction(a[v - 1])
        for child in children[v]:
            weight += calc_subtree_weight(child)
        return weight
    
    subtree_weight = {}
    for i in range(n + 1):
        subtree_weight[i] = calc_subtree_weight(i)
    
    # Calculate expected operations for each vertex
    memo = {}
    
    def expected_ops(v):
        if v in memo:
            return memo[v]
        
        # Get children with non-zero subtree weights
        child_info = []
        for child in children[v]:
            if subtree_weight[child] > 0:
                child_ops = expected_ops(child)
                child_info.append((child, subtree_weight[child], child_ops))
        
        if not child_info:
            memo[v] = Fraction(0)
            return Fraction(0)
        
        # Try all permutations of children to find the optimal order
        import itertools
        
        total_child_weight = sum(info[1] for info in child_info)
        min_cost = float('inf')
        
        for perm in itertools.permutations(child_info):
            cost = Fraction(0)
            for i, (child, child_weight, child_ops) in enumerate(perm):
                prob = child_weight / total_child_weight
                ops_to_reach = i + 1
                cost += prob * (ops_to_reach + child_ops)
            
            if cost < min_cost:
                min_cost = cost
        
        memo[v] = min_cost
        return min_cost
    
    result = expected_ops(0)
    
    # Convert to modular form
    MOD = 998244353
    def modinv(a, m=MOD):
        return pow(a, m - 2, m)
    
    numerator = result.numerator % MOD
    denominator = result.denominator % MOD
    return (numerator * modinv(denominator)) % MOD

# Main code
T = int(input())
for _ in range(T):
    print(solve_case())