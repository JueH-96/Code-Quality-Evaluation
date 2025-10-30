MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Build children list
    children = [[] for _ in range(N+1)]  # 1-based
    for i in range(1, N+1):
        a_i = A[i-1]
        if a_i == 0:
            continue
        else:
            children[a_i].append(i)
    
    # Precompute factorial and inverse factorial
    max_n = N
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    
    # Compute inverse factorial
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
    for i in range(max_n-1, 0, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Post-order processing
    s = [0] * (N + 1)
    f = [1] * (N + 1)
    
    for i in range(N, 0, -1):
        s[i] = 1
        prod_f = 1
        prod_fact = 1
        for child in children[i]:
            s[i] += s[child]
            prod_f = prod_f * f[child] % MOD
            prod_fact = prod_fact * fact[s[child]] % MOD
        numerator = fact[s[i] - 1]
        denominator = prod_fact
        if denominator == 0:
            coefficient = 0
        else:
            coefficient = numerator * pow(denominator, MOD-2, MOD) % MOD
        f[i] = prod_f * coefficient % MOD
    
    # Collect roots
    roots = []
    for i in range(1, N+1):
        if A[i-1] == 0:
            roots.append(i)
    
    product_f_roots = 1
    product_s_fact = 1
    for r in roots:
        product_f_roots = product_f_roots * f[r] % MOD
        product_s_fact = product_s_fact * fact[s[r]] % MOD
    
    answer = product_f_roots * fact[N] % MOD
    answer = answer * pow(product_s_fact, MOD-2, MOD) % MOD
    print(answer)

if __name__ == '__main__':
    main()