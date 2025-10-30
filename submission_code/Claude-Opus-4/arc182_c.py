def prime_factorization(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def count_divisors(prime_factors):
    result = 1
    for exp in prime_factors.values():
        result *= (exp + 1)
    return result

def solve(N, M):
    MOD = 998244353
    
    # Get prime factorizations of 1 to M
    factorizations = []
    for i in range(1, M + 1):
        factorizations.append(prime_factorization(i))
    
    # dp[k][state] = number of sequences of length k that produce the prime factorization state
    # state is represented as a tuple of exponents for each prime
    
    # Find all primes up to M
    primes = set()
    for f in factorizations:
        primes.update(f.keys())
    primes = sorted(primes)
    
    # Convert factorizations to tuples
    def to_tuple(factors):
        return tuple(factors.get(p, 0) for p in primes)
    
    def from_tuple(t):
        return {primes[i]: t[i] for i in range(len(primes)) if t[i] > 0}
    
    # Initialize DP
    current = {}
    for i in range(M):
        t = to_tuple(factorizations[i])
        current[t] = current.get(t, 0) + 1
    
    result = 0
    
    # Process each length
    for length in range(1, N + 1):
        # Add contribution of current length
        for state, count in current.items():
            factors = from_tuple(state)
            divisors = count_divisors(factors)
            result = (result + count * divisors) % MOD
        
        if length < N:
            # Prepare for next length
            next_dp = {}
            for state, count in current.items():
                for i in range(M):
                    new_state = tuple(state[j] + to_tuple(factorizations[i])[j] for j in range(len(primes)))
                    next_dp[new_state] = (next_dp.get(new_state, 0) + count) % MOD
            current = next_dp
    
    return result

# Read input
N, M = map(int, input().split())

# Solve and print result
print(solve(N, M))