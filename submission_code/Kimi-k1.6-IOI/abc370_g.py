MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    if N == 0:
        print(0)
        return
    
    max_fact = max(M + 33 + 100, 2 * M)
    fact = [1] * (max_fact + 2)
    for i in range(1, max_fact + 2):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (max_fact + 2)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
    
    def sieve(limit):
        sieve_list = [True] * (limit + 1)
        sieve_list[0] = sieve_list[1] = False
        for num in range(2, int(limit ** 0.5) + 1):
            if sieve_list[num]:
                sieve_list[num*num : limit+1 : num] = [False]*len(sieve_list[num*num : limit+1 : num])
        return [num for num, is_prime in enumerate(sieve_list) if is_prime]
    
    K = int(N ** 0.5)
    primes_small = sieve(K)
    
    total_sum = 1
    for p in primes_small:
        e_max = 0
        current = 1
        while current * p <= N:
            current *= p
            e_max += 1
        total_sum = total_sum * comb(e_max + M, M) % MOD
    
    def segmented_sieve(K, N):
        sieve_small = sieve(K)
        sieve_small_set = set(sieve_small)
        primes_above = []
        is_prime = [True] * (N - K) if N > K else []
        for p in sieve_small:
            if p * p > N:
                break
            start = ((K + 1) // p) * p
            if start < p * p:
                start = p * p
            if start < K + 1:
                start = max(start, K + 1)
            if start > N:
                continue
            for multiple in range(start, N + 1, p):
                if multiple - (K + 1) < 0:
                    continue
                if multiple > N:
                    break
                is_prime[multiple - (K + 1)] = False
        for q in range(K + 1, N + 1):
            if q in sieve_small_set:
                continue
            if q > N:
                break
            if (q > K and (q < K + 1 or (is_prime and is_prime[q - (K + 1)]))):
                primes_above.append(q)
        return primes_above
    
    primes_above = segmented_sieve(K, N)
    cnt_primes_above = len(primes_above)
    if primes_above:
        total_sum = total_sum * pow((M + 1) % MOD, cnt_primes_above, MOD) % MOD
    
    sum_3 = 0
    k_max_3 = 0
    current = 1
    while current <= N:
        sum_3 = comb(k_max_3 + M, M)
        next_current = current * 3
        if next_current > N:
            break
        current = next_current
        k_max_3 += 1
    
    product_P = 1
    for p in primes_small:
        if p == 3:
            continue
        e_max = 0
        current = 1
        while current <= N:
            if current * p > N:
                break
            current *= p
            e_max += 1
        s = 0
        if p % 3 == 1:
            e = 0
            while True:
                if pow(p, e, MOD) > N:
                    break
                if e % 3 == 0 or e % 3 == 1:
                    s = (s + comb(e + M - 1, M - 1)) % MOD
                e += 1
                if e > e_max:
                    break
        else:
            e = 0
            while e <= e_max:
                if e % 2 == 0:
                    s = (s + comb(e + M - 1, M - 1)) % MOD
                e += 1
        product_P = product_P * s % MOD
    
    primes_above_filtered = [p for p in primes_above if p != 3]
    cnt_primes_above_equiv1 = 0
    cnt_primes_above_equiv2 = 0
    for p in primes_above_filtered:
        if p % 3 == 1:
            cnt_primes_above_equiv1 += 1
        else:
            cnt_primes_above_equiv2 += 1
    
    contribution_above = pow((M + 1) % MOD, cnt_primes_above_equiv1, MOD)
    product_P = product_P * contribution_above % MOD
    
    sum_3 = comb(k_max_3 + M, M)
    bad_sum = sum_3 * product_P % MOD
    
    ans = (total_sum - bad_sum) % MOD
    if ans < 0:
        ans += MOD
    print(ans)

if __name__ == "__main__":
    main()