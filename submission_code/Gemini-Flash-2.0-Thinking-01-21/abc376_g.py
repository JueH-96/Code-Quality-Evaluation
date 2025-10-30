def solve():
    n = int(input())
    parents = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    total_a = sum(a)
    if total_a == 0:
        print(0)
        return
        
    probabilities = [0] * (n + 1)
    for i in range(n):
        probabilities[i+1] = a[i]
        
    children = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        children[parents[i-1]].append(i)
        
    P = [0] * (n + 1)
    E = [0] * (n + 1)
    
    mod = 998244353
    
    def power(base, exp):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return res
        
    def mod_inverse(n):
        return power(n, mod - 2)
        
    total_a_mod_inv = mod_inverse(total_a)
    prob_values = [0] * (n + 1)
    for i in range(1, n + 1):
        prob_values[i] = (probabilities[i] * total_a_mod_inv) % mod
        
    memo_pe = {}
    
    def get_P_E(u):
        if u in memo_pe:
            return memo_pe[u]
        
        current_P = prob_values[u]
        child_probabilities_expected_values = []
        for v in children[u]:
            child_p, child_e = get_P_E(v)
            current_P = (current_P + child_p) % mod
            child_probabilities_expected_values.append({'vertex': v, 'P': child_p, 'E': child_e})
            
        sorted_children = sorted(child_probabilities_expected_values, key=lambda x: x['P'], reverse=True)
        
        current_E = 1
        sum_weighted_e = 0
        for child_info in sorted_children:
            sum_weighted_e = (sum_weighted_e + (child_info['P'] * child_info['E']) % mod) % mod
            
        if current_P != 0:
            current_E = (current_E + (sum_weighted_e * mod_inverse(current_P)) % mod) % mod
        else:
            current_E = 1
            
        result = (current_P, current_E)
        memo_pe[u] = result
        return result
        
    for i in range(1, n + 1):
        if i not in memo_pe:
            get_P_E(i)
            
    root_children = children[0]
    root_child_probabilities_expected_values = []
    for child_vertex in root_children:
        p_val, e_val = memo_pe[child_vertex]
        root_child_probabilities_expected_values.append({'vertex': child_vertex, 'P': p_val, 'E': e_val})
        
    sorted_root_children = sorted(root_child_probabilities_expected_values, key=lambda x: x['P'], reverse=True)
    
    expected_operations = 0
    for index, child_info in enumerate(sorted_root_children):
        rank = index + 1
        term1 = (rank * child_info['P']) % mod
        term2 = (child_info['P'] * child_info['E']) % mod
        expected_operations = (expected_operations + term1 + term2) % mod
        
    print(expected_operations)

t = int(input())
for _ in range(t):
    solve()