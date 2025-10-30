import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_A = max(A) if A else 0
    max_A = max(max_A, 10**5)
    
    # Precompute divisors for all numbers up to max_A
    divisors = [[] for _ in range(max_A + 1)]
    for i in range(1, max_A + 1):
        for j in range(i, max_A + 1, i):
            divisors[j].append(i)
    
    # Precompute Möbius function
    max_mobius = max_A
    mobius = [1] * (max_mobius + 1)
    is_prime = [True] * (max_mobius + 1)
    for p in range(2, max_mobius + 1):
        if is_prime[p]:
            for multiple in range(p, max_mobius + 1, p):
                is_prime[multiple] = False if multiple != p else is_prime[multiple]
                mobius[multiple] *= -1
            p_square = p * p
            for multiple in range(p_square, max_mobius + 1, p_square):
                mobius[multiple] = 0
    
    # Precompute pow_2 and inv_2
    inv_2 = pow(2, MOD-2, MOD)
    pow_2 = [1] * (N + 2)
    inv_2_pows = [1] * (N + 2)
    for i in range(1, N + 2):
        pow_2[i] = pow_2[i-1] * 2 % MOD
        inv_2_pows[i] = inv_2_pows[i-1] * inv_2 % MOD
    
    sum_g = dict()
    prefix_sum = [0] * (N + 1)
    
    for j in range(1, N + 1):
        a = A[j-1]
        s_j = 0
        for d in divisors[a]:
            if d == 0:
                continue
            k = a // d
            temp = 0
            for t in divisors[k]:
                mu = mobius[t]
                if mu == 0:
                    continue
                g = d * t
                sum_g_val = sum_g.get(g, 0)
                temp = (temp + mu * sum_g_val) % MOD
            s_j = (s_j + d * temp) % MOD
        
        term_j = s_j * inv_2_pows[j] % MOD
        prefix_sum[j] = (prefix_sum[j-1] + term_j) % MOD
        
        pow_2_j_minus_1 = pow(2, j-1, MOD)
        for g in divisors[a]:
            sum_g[g] = (sum_g.get(g, 0) + pow_2_j_minus_1) % MOD
    
    for m in range(1, N + 1):
        ans = prefix_sum[m] * pow_2[m] % MOD
        print(ans)

if __name__ == '__main__':
    main()