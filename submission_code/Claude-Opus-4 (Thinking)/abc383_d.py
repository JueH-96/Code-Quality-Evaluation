def sieve_of_eratosthenes(limit):
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]

N = int(input())
count = 0

# Case 1: p^8
limit1 = int(N ** 0.125)
# Check if we need to increase limit1 due to floating point precision
if (limit1 + 1) ** 8 <= N:
    limit1 += 1

primes1 = sieve_of_eratosthenes(limit1)
for p in primes1:
    if p ** 8 <= N:
        count += 1

# Case 2: p^2 * q^2 where p < q
sqrt_N = int(N ** 0.5)
# Check if we need to increase sqrt_N due to floating point precision
if (sqrt_N + 1) ** 2 <= N:
    sqrt_N += 1

primes2 = sieve_of_eratosthenes(sqrt_N)

for i in range(len(primes2)):
    p = primes2[i]
    for j in range(i+1, len(primes2)):
        q = primes2[j]
        if p * q > sqrt_N:
            break
        count += 1

print(count)