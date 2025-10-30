def count_numbers_with_9_divisors(N):
    count = 0
    
    # Case 1: n = p^8
    # We need p^8 <= N, so p <= N^(1/8)
    limit = int(N ** (1/8)) + 1
    
    # Check each potential prime p
    for p in range(2, limit + 1):
        if is_prime(p):
            if p ** 8 <= N:
                count += 1
    
    # Case 2: n = p^2 * q^2 where p < q
    # We need p^2 * q^2 <= N
    # Since p < q, we have p^2 < pq, so p^2 < sqrt(N)
    # Thus p < N^(1/4)
    
    primes = []
    limit2 = int(N ** 0.5) + 1
    
    # Generate all primes up to sqrt(N)
    for i in range(2, limit2 + 1):
        if is_prime(i):
            primes.append(i)
    
    # Count pairs (p, q) where p < q and p^2 * q^2 <= N
    for i in range(len(primes)):
        p = primes[i]
        if p * p > N ** 0.5:
            break
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if p * p * q * q > N:
                break
            count += 1
    
    return count

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Read input
N = int(input())

# Calculate and print the answer
print(count_numbers_with_9_divisors(N))