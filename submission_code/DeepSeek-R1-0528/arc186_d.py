import sys

mod = 998244353

def precompute_factorials(n, mod):
    fact = [1] * (n+1)
    inv_fact = [1] * (n+1)
    for i in range(2, n+1):
        fact[i] = fact[i-1] * i % mod
    inv_fact[n] = pow(fact[n], mod-2, mod)
    for i in range(n, 0, -1):
        inv_fact[i-1] = inv_fact[i] * i % mod
    return fact, inv_fact

def precompute_catalan(n, mod):
    cat = [0] * (n+1)
    if n >= 0:
        cat[0] = 1
    for i in range(1, n+1):
        cat[i] = (2 * (2*i - 1) * cat[i-1]) % mod
        cat[i] = cat[i] * pow(i+1, mod-2, mod) % mod
    return cat

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print(0)
        return
        
    if n > 1000:
        max_catalan = n
        cat = precompute_catalan(max_catalan, mod)
        print(cat[n-1])
        return
        
    max_fact = 2 * n + 10
    if max_fact < 600000:
        max_fact = 600000
    fact, inv_fact = precompute_factorials(max_fact, mod)
    
    def nCr(n_val, r_val):
        if r_val < 0 or r_val > n_val:
            return 0
        return fact[n_val] * inv_fact[r_val] % mod * inv_fact[n_val - r_val] % mod
        
    def T_tree(v, l):
        L = l - 1
        if L < v:
            return 0
        n_val = 2 * L - v - 1
        r_val = L - v
        if n_val < 0 or r_val < 0 or r_val > n_val:
            return 0
        comb = nCr(n_val, r_val)
        res = v * pow(L, mod-2, mod) % mod * comb % mod
        return res

    def T_forest(k, L_val):
        if L_val == 0:
            return 1 if k == 0 else 0
        if k == 0:
            return 0
        if L_val < k:
            return 0
        n_val = 2 * L_val - k - 1
        r_val = L_val - k
        if n_val < 0 or r_val < 0 or r_val > n_val:
            return 0
        comb = nCr(n_val, r_val)
        res = k * pow(L_val, mod-2, mod) % mod * comb % mod
        return res

    memo_tree = {}
    memo_forest = {}
    
    def dfs_tree(start, l):
        if (start, l) in memo_tree:
            return memo_tree[(start, l)]
        if start + l > n:
            return (0, False)
        if l == 1:
            if A[start] < 0:
                res = (0, False)
            else:
                res = (1, A[start] == 0)
            memo_tree[(start, l)] = res
            return res
            
        count = 0
        overall_valid = False
        if A[start] > 0:
            v_min = 1
            v_max = min(A[start] - 1, l - 1)
            if v_min <= v_max:
                for v in range(v_min, v_max + 1):
                    term = T_tree(v, l)
                    count = (count + term) % mod
        if A[start] >= 1 and A[start] <= l - 1:
            v = A[start]
            cnt_forest, valid_forest = dfs_forest(start + 1, l - 1, v)
            count = (count + cnt_forest) % mod
            if valid_forest:
                overall_valid = True
        res = (count, overall_valid)
        memo_tree[(start, l)] = res
        return res

    def dfs_forest(start, l, k):
        key = (start, l, k)
        if key in memo_forest:
            return memo_forest[key]
        if start + l > n:
            return (0, False)
        if k == 0:
            return (1, True) if l == 0 else (0, False)
        count = 0
        overall_valid = False
        min_rest = k - 1
        for l1 in range(1, l - min_rest + 1):
            if start + l1 > n:
                break
            cnt1, valid1 = dfs_tree(start, l1)
            cnt_rest_arbitrary = T_forest(k - 1, l - l1)
            cnt_less1 = cnt1
            if valid1:
                cnt_less1 = (cnt1 - 1) % mod
            option = cnt_less1 * cnt_rest_arbitrary % mod
            if valid1:
                cnt_rest_lex, valid_rest = dfs_forest(start + l1, l - l1, k - 1)
                option = (option + cnt_rest_lex) % mod
                if valid_rest:
                    if not overall_valid:
                        overall_valid = True
            count = (count + option) % mod
        res = (count, overall_valid)
        memo_forest[key] = res
        return res

    total, _ = dfs_tree(0, n)
    print(total % mod)

if __name__ == '__main__':
    main()