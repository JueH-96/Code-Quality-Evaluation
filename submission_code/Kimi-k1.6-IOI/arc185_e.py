MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if N == 0:
        return
    
    max_A = max(A)
    
    # Precompute smallest prime factors (SPF)
    spf = list(range(max_A + 1))
    for i in range(2, int(max_A**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_A + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute phi
    phi = [0] * (max_A + 1)
    phi[1] = 1
    for n in range(2, max_A + 1):
        if spf[n] == n:  # n is prime
            phi[n] = n - 1
        else:
            primes = set()
            temp = n
            while temp != 1:
                p = spf[temp]
                primes.add(p)
                temp //= p
            res = n
            for p in primes:
                res = res // p * (p - 1)
            phi[n] = res
    
    # Precompute divisors
    divisors = [[] for _ in range(max_A + 1)]
    for d in range(1, max_A + 1):
        for m in range(d, max_A + 1, d):
            divisors[m].append(d)
    
    # Precompute pow2
    pow2 = [1] * N
    for i in range(1, N):
        pow2[i] = (pow2[i-1] * 2) % MOD
    
    # Initialize C and sum_ans
    C = [0] * (max_A + 1)
    sum_ans = 0
    
    for i in range(N):
        x = A[i]
        # Compute G_i
        g = 0
        for d in divisors[x]:
            g += phi[d] * C[d]
            if g >= MOD:
                g -= MOD
        sum_ans = (2 * sum_ans + g) % MOD
        print(sum_ans)
        # Update C
        add = pow2[i]
        for d in divisors[x]:
            C[d] = (C[d] + add) % MOD

if __name__ == "__main__":
    main()