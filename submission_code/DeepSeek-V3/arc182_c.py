# YOUR CODE HERE
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    # Precompute the number of divisors for each possible product
    # Since M is small (<=16), we can precompute the number of divisors for all possible products
    # The maximum product is M^N, but since N can be up to 1e18, we need a smarter approach
    
    # Instead, we can precompute the number of divisors for each number from 1 to M^N
    # But since N can be up to 1e18, this is not feasible
    
    # Alternative approach: use the fact that the number of divisors is multiplicative
    # For each sequence, the product X is the product of the elements, and the number of divisors is the product of (e_i + 1) for each prime factor p_i^e_i in X
    
    # Since M is small, we can precompute the prime factors for each number from 1 to M
    # Then, for each sequence, we can compute the exponents of the prime factors in the product X
    
    # Precompute the prime factors for each number from 1 to M
    primes = []
    for i in range(2, M+1):
        is_prime = True
        for p in primes:
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    # Precompute the exponents of each prime in the factorization of each number from 1 to M
    factorizations = {}
    for num in range(1, M+1):
        factors = {}
        temp = num
        for p in primes:
            if p*p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp = temp // p
        if temp != 1:
            factors[temp] = 1
        factorizations[num] = factors
    
    # Now, for each sequence, the product X's factorization is the sum of the factorizations of its elements
    # The number of divisors is the product of (e_i + 1) for each prime factor p_i^e_i in X
    
    # To compute the sum of the number of divisors for all sequences, we can use dynamic programming
    # Let dp[k][e1][e2]...[en] be the number of sequences of length k where the exponents of the primes are e1, e2, ..., en
    # Then, the number of divisors for such a sequence is (e1+1)*(e2+1)*...*(en+1)
    
    # However, since the number of primes is small (up to 5 for M=16), we can manage this
    
    # Initialize the dp table
    # We will represent the exponents as a tuple
    # Initialize with sequences of length 1
    dp = {}
    for num in range(1, M+1):
        factors = factorizations[num]
        key = tuple(sorted(factors.items()))
        dp[key] = dp.get(key, 0) + 1
    
    # Now, for each length from 2 to N, update the dp table
    for _ in range(2, N+1):
        new_dp = {}
        for key in dp:
            for num in range(1, M+1):
                factors = factorizations[num]
                new_key = list(key)
                for p, e in factors.items():
                    found = False
                    for i in range(len(new_key)):
                        if new_key[i][0] == p:
                            new_key[i] = (p, new_key[i][1] + e)
                            found = True
                            break
                    if not found:
                        new_key.append((p, e))
                new_key = tuple(sorted(new_key))
                new_dp[new_key] = (new_dp.get(new_key, 0) + dp[key]) % MOD
        dp = new_dp
    
    # Now, compute the sum of the number of divisors for all sequences
    total = 0
    for key in dp:
        divisors = 1
        for p, e in key:
            divisors *= (e + 1)
        total = (total + divisors * dp[key]) % MOD
    
    print(total)

if __name__ == "__main__":
    main()