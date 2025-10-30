# YOUR CODE HERE
import math
from collections import defaultdict

def get_coprime_factorizations(n):
    """Get all ways to factor n as p*q where gcd(p,q)=1"""
    factors = []
    for p in range(1, int(n**0.5) + 1):
        if n % p == 0:
            q = n // p
            if math.gcd(p, q) == 1:
                factors.append((p, q))
                if p != q:
                    factors.append((q, p))
    return factors

def get_prime_factors(n):
    """Get prime factorization of n"""
    factors = defaultdict(int)
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] += 1
            n //= d
        d += 1
    if n > 1:
        factors[n] += 1
    return factors

def solve():
    MOD = 998244353
    
    N = int(input())
    A = list(map(int, input().split()))
    
    # Get all coprime factorizations for each A_i
    factorizations = []
    for a in A:
        factorizations.append(get_coprime_factorizations(a))
    
    # Find all primes that appear in any A_i
    all_primes = set()
    for a in A:
        pf = get_prime_factors(a)
        all_primes.update(pf.keys())
    all_primes = sorted(list(all_primes))
    
    # Use inclusion-exclusion to handle gcd constraint
    total = 0
    
    # Iterate over all subsets of primes to exclude
    for mask in range(1 << len(all_primes)):
        excluded_primes = []
        for i in range(len(all_primes)):
            if mask & (1 << i):
                excluded_primes.append(all_primes[i])
        
        # Calculate contribution with these primes excluded
        # dp[i][constraint] = sum of products of first i elements
        # constraint tracks which factorizations are valid
        
        valid = True
        choices = []
        
        for i in range(N-1):
            valid_facts = []
            for p, q in factorizations[i]:
                # Check if this factorization is valid given excluded primes
                valid_for_excluded = True
                for prime in excluded_primes:
                    if p % prime == 0 or q % prime == 0:
                        valid_for_excluded = False
                        break
                if valid_for_excluded:
                    valid_facts.append((p, q))
            
            if not valid_facts:
                valid = False
                break
            choices.append(valid_facts)
        
        if not valid:
            continue
        
        # Dynamic programming to count sequences
        # dp[i] = map from divisibility constraints to (count, sum of products)
        dp = [defaultdict(lambda: [0, 0]) for _ in range(N+1)]
        
        # Initialize
        dp[0][(1,)] = [1, 1]
        
        for i in range(N-1):
            for state, (cnt, prod_sum) in dp[i].items():
                if cnt == 0:
                    continue
                
                for p, q in choices[i]:
                    # Update state based on constraints
                    new_state = list(state)
                    
                    if i == 0:
                        new_state = [p]
                    
                    # The i-th element must be divisible by new_state[-1]
                    # The (i+1)-th element must be divisible by q
                    
                    if i < N-2:
                        # Need to track constraint for next position
                        lcm_val = (new_state[-1] * q) // math.gcd(new_state[-1], q)
                        new_state = tuple([lcm_val])
                    else:
                        new_state = tuple([q])
                    
                    # Update dp
                    if i == 0:
                        # First element value is p * (some multiplier)
                        for mult in range(1, 1001):
                            if all(mult * p % prime != 0 for prime in excluded_primes):
                                val = mult * p
                                dp[i+1][new_state][0] = (dp[i+1][new_state][0] + cnt) % MOD
                                dp[i+1][new_state][1] = (dp[i+1][new_state][1] + prod_sum * val) % MOD
                    else:
                        # Current element must be divisible by state[-1] and p
                        lcm_constraint = (state[-1] * p) // math.gcd(state[-1], p)
                        for mult in range(1, 1001):
                            if all(mult * lcm_constraint % prime != 0 for prime in excluded_primes):
                                val = mult * lcm_constraint
                                dp[i+1][new_state][0] = (dp[i+1][new_state][0] + cnt) % MOD
                                dp[i+1][new_state][1] = (dp[i+1][new_state][1] + prod_sum * val) % MOD
        
        # Last element
        contribution = 0
        for state, (cnt, prod_sum) in dp[N-1].items():
            if cnt == 0:
                continue
            # Last element must be divisible by state[-1]
            for mult in range(1, 1001):
                if all(mult * state[-1] % prime != 0 for prime in excluded_primes):
                    val = mult * state[-1]
                    contribution = (contribution + prod_sum * val) % MOD
        
        # Apply inclusion-exclusion
        if len(excluded_primes) % 2 == 0:
            total = (total + contribution) % MOD
        else:
            total = (total - contribution) % MOD
    
    print(total)

solve()