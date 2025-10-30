import sys

# Fast I/O
input = sys.stdin.readline
print = sys.stdout.write

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    MOD = 998244353
    MAX_A_VAL = 100000

    # 1. Precompute phi values (Euler's totient function)
    # phi[i] stores the count of numbers up to i that are relatively prime to i.
    phi = [0] * (MAX_A_VAL + 1)
    for i in range(MAX_A_VAL + 1):
        phi[i] = i
    for i in range(2, MAX_A_VAL + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, MAX_A_VAL + 1, i):
                phi[j] -= phi[j] // i

    # 2. Precompute smallest prime factor (spf) for efficient divisor generation
    # spf[i] stores the smallest prime factor of i.
    spf = [0] * (MAX_A_VAL + 1)
    primes = []
    for i in range(2, MAX_A_VAL + 1):
        if spf[i] == 0: # i is prime
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > MAX_A_VAL:
                break
            spf[i * p] = p

    # Function to get all divisors of a number n
    # Uses precomputed spf to get prime factorization, then generates divisors recursively.
    def get_divisors(n):
        prime_factorization = []
        temp_n = n
        while temp_n > 1:
            p = spf[temp_n]
            count = 0
            while temp_n % p == 0:
                temp_n //= p
                count += 1
            prime_factorization.append((p, count))
        
        divs = []
        def generate(idx, current_d):
            if idx == len(prime_factorization):
                divs.append(current_d)
                return
            
            p, e = prime_factorization[idx]
            p_power = 1
            for _ in range(e + 1): # Loop from 0th power to eth power of p
                generate(idx + 1, current_d * p_power)
                p_power *= p # Move to next power of p
        
        generate(0, 1)
        return divs

    # 3. Precompute powers of 2 modulo MOD
    pow2 = [1] * N
    for i in range(1, N):
        pow2[i] = (pow2[i-1] * 2) % MOD

    # Initialize dp_sum_scores (S(m)) and term_val array
    dp_sum_scores = 0 
    term_val = [0] * (MAX_A_VAL + 1) # term_val[v] = sum of 2^k for A[k] where v | A[k] and k < current_i

    # Main loop: Iterate through the sequence A
    for i in range(N):
        current_A = A[i]
        
        # Calculate f_current_A = sum_{v | current_A} phi(v) * term_val[v]
        # This is the sum of gcd(last(B), current_A) terms, weighted by powers of 2.
        # last(B) corresponds to A[k] for k < i.
        f_current_A = 0
        divisors_of_A_i = get_divisors(current_A)
        
        for d in divisors_of_A_i:
            f_current_A = (f_current_A + phi[d] * term_val[d]) % MOD
        
        # Update dp_sum_scores: S(i+1) = (2 * S(i) + f_current_A) % MOD
        # Here dp_sum_scores holds S(i) from the previous iteration.
        dp_sum_scores = (2 * dp_sum_scores + f_current_A) % MOD
        
        # Print the result for m = i+1
        sys.stdout.write(str(dp_sum_scores) + "
")
        
        # Update term_val for the current A[i] for future calculations (for A[k] where k > i)
        # Add pow2[i] (which is 2^i) to term_val[d] for all divisors d of A[i].
        for d in divisors_of_A_i:
            term_val[d] = (term_val[d] + pow2[i]) % MOD

solve()