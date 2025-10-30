import sys
MOD = 998244353

def main():
    input = sys.stdin.read().strip()
    N = int(input)
    
    # Precompute digit counts
    d = [0] * (N + 1)
    for x in range(1, N + 1):
        temp = x
        cnt = 0
        while temp > 0:
            temp //= 10
            cnt += 1
        d[x] = cnt
    
    # Precompute fact and inv_fact
    max_n = N
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    
    # Group by d
    from collections import defaultdict
    d_groups = defaultdict(list)
    for x in range(1, N+1):
        d_groups[d[x]].append(x)
    
    # Precompute for each d: cnt_d, sum_d, a_d
    d_list = []
    for key in sorted(d_groups.keys()):
        xs = d_groups[key]
        cnt_d = len(xs)
        sum_d = sum(xs)
        # a_d = 1 / 10^key mod MOD
        pow10 = pow(10, key, MOD)
        a_d = pow(pow10, MOD-2, MOD)
        d_list.append( (cnt_d, a_d) )
    
    # Compute S(x) = product_d (1 + a_d x)^{cnt_d}
    # Initial S is [1]
    S = [0] * (N + 1)
    S[0] = 1
    for cnt_d, a_d in d_list:
        # Compute (1 + a_d x)^{cnt_d} and multiply with S
        # Create a temporary polynomial for (1 + a_d x)^{cnt_d}
        temp_poly = [0] * (min(cnt_d, N) + 1)
        for k in range(min(cnt_d, N) + 1):
            # C(cnt_d, k) * a_d^k
            if k > cnt_d:
                break
            # Compute combination
            if k > cnt_d:
                C = 0
            else:
                C = fact[cnt_d] * pow(fact[k], MOD-2, MOD) % MOD
                C = C * pow(fact[cnt_d - k], MOD-2, MOD) % MOD
            a_power = pow(a_d, k, MOD)
            temp_poly[k] = C * a_power % MOD
        
        # Multiply S with temp_poly
        new_S = [0] * (len(S) + len(temp_poly) - 1)
        for i in range(len(S)):
            if S[i] == 0:
                continue
            for j in range(len(temp_poly)):
                if i + j >= len(new_S):
                    break
                new_S[i + j] = (new_S[i + j] + S[i] * temp_poly[j]) % MOD
        # Truncate to N+1
        S = new_S[:N+1]
    
    # Compute D_m for m from 0 to N-1
    D = [0] * N
    for m in range(N):
        dm = 0
        for x in range(1, N+1):
            power = m + 1
            dx = d[x]
            # 1/10^{dx * (m+1)} = (1/10^{dx})^{m+1}
            pow10 = pow(10, dx, MOD)
            inv_pow10 = pow(pow10, MOD-2, MOD)
            a_power = pow(inv_pow10, power, MOD)
            term = x * a_power % MOD
            dm = (dm + term) % MOD
        D[m] = dm
    
    # Compute A_t for t from 0 to N-1
    A = [0] * N
    for t in range(N):
        at = 0
        for m in range(t+1):
            sign = -1 if m % 2 else 1
            if sign == -1:
                C_term = (MOD - S[t - m]) % MOD
            else:
                C_term = S[t - m] % MOD
            term = C_term * D[m] % MOD
            at = (at + term) % MOD
        A[t] = at
    
    # Compute the total
    total_digits = sum(d[1:N+1])
    pow10T = pow(10, total_digits, MOD)
    
    res = 0
    for i in range(1, N+1):
        t = i-1
        if t >= N:
            continue
        term = fact[t] * fact[N-1 - t] % MOD
        term = term * A[t] % MOD
        res = (res + term) % MOD
    
    res = res * pow10T % MOD
    print(res)

if __name__ == "__main__":
    main()