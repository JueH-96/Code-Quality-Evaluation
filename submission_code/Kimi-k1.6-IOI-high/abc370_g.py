import sys
import math

MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())

    if N ==0:
        print(0)
        return

    max_fact = M + 40  # A safe upper bound for combinations
    fact = [1] * (max_fact +1)
    for i in range(1, max_fact+1):
        fact[i] = fact[i-1] * i % MOD

    inv_fact = [1]*(max_fact +1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    def comb(n, k):
        if n <0 or k <0 or n <k:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n -k] % MOD

    sqrt_N = int(math.isqrt(N))
    sieve = [True] * (sqrt_N +1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(sqrt_N)) +1):
        if sieve[i]:
            sieve[i*i : sqrt_N+1 : i] = [False]*len(sieve[i*i : sqrt_N+1 : i])
    sieve_primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    # Sieve of Eratosthenes for sieve_primes
    primes = sieve_primes.copy()
    # Generate primes in (sqrt_N, N] using segmented sieve
    def segmented_sieve(L, R):
        limit = R
        sqrt_limit = int(math.isqrt(limit)) +1
        sieve_small = [True] * (sqrt_limit +1)
        sieve_small[0] = sieve_small[1] = False
        for i in range(2, int(math.isqrt(sqrt_limit)) +1):
            if sieve_small[i]:
                sieve_small[i*i : sqrt_limit+1 : i] = [False]*len(sieve_small[i*i : sqrt_limit+1 : i])
        primes_small = [i for i, is_p in enumerate(sieve_small) if is_p]

        is_prime = [True]*(R-L+1)
        if L ==0:
            is_prime[0] = False
        for p in primes_small:
            start = L // p * p
            if start < L:
                start += p
            if start == p:
                start += p
            for multiple in range(start, R+1, p):
                is_prime[multiple - L] = False
        primes_segment = []
        for num in range(L, R+1):
            if is_prime[num - L]:
                primes_segment.append(num)
        return primes_segment

    L = sqrt_N +1
    R = N
    primes_high = []
    if L <= R:
        primes_high = segmented_sieve(L, R)

    total_sum_primes = []

    # Process sieve_primes (primes up to sqrt(N))
    total_product = 1
    bad_product = 1
    for p in sieve_primes:
        e_p = 0
        current = 1
        while current <= N:
            current *= p
            e_p +=1
        e_p -=1  # adjust e_p as compute_e_p returns floor(log_p N)
        sum_total = comb(e_p + M, M)
        total_product = total_product * sum_total % MOD

        mod3 = p %3
        sum_bad =0
        if mod3 ==0:
            # p is 3
            sum_bad = comb(e_p + M, M)
        elif mod3 ==1:
            # allowed k: 0 mod3 or 1 mod3
            sum_bad =0
            for k in range(0, e_p+1):
                if k %3 ==0 or k%3 ==1:
                    sum_bad += comb(k + M -1, M-1)
                    sum_bad %= MOD
        else:
            # mod3 ==2: allowed even k
            sum_bad =0
            for k in range(0, e_p+1, 2):
                sum_bad += comb(k + M -1, M-1)
                sum_bad %= MOD
        bad_product = bad_product * sum_bad % MOD

    # Process primes_high (primes in (sqrt_N, N])
    count_high_total = len(primes_high)
    # count_high_bad: number of primes which are 1 mod3 (sum_bad contribution is M+1) and 2 mod3 (sum_bad contribution is 1)
    count_1_mod3 =0
    count_2_mod3 =0
    count_3 =0
    for p in primes_high:
        if p ==3:
            count_3 +=1
        else:
            mod3 = p%3
            if mod3 ==1:
                count_1_mod3 +=1
            else:
                count_2_mod3 +=1

    # Handle contribution from primes_high for total_sum
    term_total = comb(1 + M, M)
    total_product_high = pow(term_total, count_high_total, MOD)
    total_product = total_product * total_product_high % MOD

    # For bad_sum, each prime_high contributes as follows:
    # p ==1 mod3: sum_bad is comb(1 + M, M) = M+1 choose M = M+1
    # p ==2 mod3: sum_bad is comb(0 + M-1, M-1) =1
    # p ==3: sum_bad is comb(1 + M, M)
    term_bad_1_mod3 = comb(1 + M, M)
    term_bad_2_mod3 = comb(0 + M-1, M-1)
    term_bad_3 = comb(1 + M, M)

    # Compute bad_sum contribution from primes_high
    total_primes_high = count_1_mod3 + count_2_mod3 + count_3
    # Check if primes_high includes 3
    # Example, for N=3, primes_high is empty. For N=4, primes_high may include 3
    # Wait, primes_high are primes between sqrt_N+1 and N. So if N >=3 and sqrt_N+1 <=3 <=N:
    # primes_high includes 3 only if sqrt_N <3 <=N. Which is true when N >=3 and sqrt(N) <3 → N <9.
    # But in segmented sieve code, if L <=3 <=R, then primes_high will include 3.
    # But in this case, p mod3 is 0, which is handled in the sieve_primes part if p <=sqrt_N.
    # So in the code above, primes_high for p==3 should not be possible unless L <=3 <=R.
    # So in the code above, primes_high is generated between L and R, which is (sqrt_N+1, N]. So if sqrt_N+1 <=3 <=N:
    # Then primes_high will include 3.
    # Example: N=4, sqrt_N=2, sieve_primes are 2. primes_high is 3 (if generated).
    # Then, in sieve_primes, 3 is not included (sqrt_N is 2), but primes_high includes 3.

    # So in the code above, primes_high may include 3, which has mod3==0.

    # So the count_3 is the number of primes_high that are 3. But since primes_high are primes between sqrt_N+1 and N, which can include 3 only when sqrt_N+1 <=3 <=N. For sqrt_N=2, primes_high starts at3.

    bad_product_high = (pow(term_bad_1_mod3, count_1_mod3, MOD) *
                       pow(term_bad_2_mod3, count_2_mod3, MOD) %
                       MOD)
    # Handle primes_high p==3 (mod3==0)
    if count_3 >0:
        bad_product_high = bad_product_high * pow(term_bad_3, count_3, MOD) % MOD

    bad_product = bad_product * bad_product_high % MOD

    answer = (total_product - bad_product) % MOD
    print(answer if answer >=0 else answer + MOD)

if __name__ == "__main__":
    main()