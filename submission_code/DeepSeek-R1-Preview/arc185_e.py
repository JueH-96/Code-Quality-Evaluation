import sys
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute powers of 2 modulo MOD
    pow2 = [1] * (N)
    for i in range(1, N):
        pow2[i] = (pow2[i-1] * 2) % MOD
    
    # F is an array where F[d] holds the sum of 2^{i-1} for i where A[i] is divisible by d
    max_A = 10**5
    F = [0] * (max_A + 2)
    
    S_prev = 0
    
    for m in range(1, N+1):
        a_m = A[m-1]
        # Compute all divisors of a_m
        divs = set()
        for i in range(1, int(a_m**0.5) + 1):
            if a_m % i == 0:
                divs.add(i)
                divs.add(a_m // i)
        divs = sorted(divs, reverse=True)
        
        # Compute S for each d
        S = {}
        for d in divs:
            sum_k = 0
            for k in divs:
                if k > d and k % d == 0:
                    sum_k += S.get(k, 0)
            sum_k %= MOD
            S[d] = (F[d] - sum_k) % MOD
        
        # Compute C(m)
        C = 0
        for d in divs:
            C = (C + d * S[d]) % MOD
        
        # Compute S_current
        S_current = (2 * S_prev + C) % MOD
        print(S_current)
        
        # Update F for each divisor of a_m
        current_pow = pow2[m-1]
        for d in divs:
            F[d] = (F[d] + current_pow) % MOD
        
        S_prev = S_current

if __name__ == '__main__':
    main()