def count_divisors(n):
    MOD = 998244353
    primes = [2, 3, 5, 7, 11, 13]
    prime_counts = [0] * len(primes)
    
    for i, p in enumerate(primes):
        while n % p == 0:
            prime_counts[i] += 1
            n //= p
            
    if n > 1:
        return 0  # Number has prime factor > 13
        
    result = 1
    for count in prime_counts:
        result *= (count + 1)
    return result

def solve():
    MOD = 998244353
    N, M = map(int, input().split())
    
    # For each number 1..M, calculate its prime factorization
    factorizations = []
    for i in range(1, M+1):
        factorizations.append(count_divisors(i))
    
    # Calculate sum of scores for sequences of each length k from 1 to N
    total = 0
    curr_pow = 1
    for k in range(1, min(N+1, 64)):
        subtotal = 0
        for i in range(M):
            subtotal = (subtotal + factorizations[i] * curr_pow) % MOD
        curr_pow = (curr_pow * M) % MOD
        total = (total + subtotal) % MOD
        
    print(total)

solve()