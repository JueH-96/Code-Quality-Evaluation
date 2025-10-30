from collections import Counter

def odd_prime_product(n):
    if n == 0:
        return 0  # Special handling for 0

    result = 1
    i = 2
    while i * i <= n:
        count = 0
        while n % i == 0:
            n //= i
            count += 1
        if count % 2 == 1:
            result *= i
        i += 1
    if n > 1:  # If n is a prime or has a prime factor > sqrt(n)
        result *= n
    return result

N = int(input())
A = list(map(int, input().split()))

# Count zero elements
zero_count = sum(1 for a in A if a == 0)
non_zero_count = N - zero_count

# Count pairs involving at least one zero
count = 0
count += zero_count * non_zero_count  # One zero, one non-zero
count += zero_count * (zero_count - 1) // 2  # Both zeros

# Count pairs of non-zero elements with the same odd_prime_product
odd_primes = [odd_prime_product(a) for a in A if a != 0]
groups = Counter(odd_primes)
for op, freq in groups.items():
    count += freq * (freq - 1) // 2

print(count)