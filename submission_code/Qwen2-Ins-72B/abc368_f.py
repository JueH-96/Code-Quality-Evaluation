from collections import Counter

N = int(input())
A = list(map(int, input().split()))

def factorize(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
    if n > 1:
        factors.append(n)
    return factors

# Factorize each element and count occurrences of each factor
factors = [factorize(a) for a in A]
factor_counts = Counter()
for f in factors:
    for prime in set(f):
        factor_counts[prime] += 1

# Count the number of piles for each prime factor
piles = Counter()
for prime, count in factor_counts.items():
    if count > 0:
        piles[count] += 1

# Calculate the nim sum
nim_sum = 0
for pile, count in piles.items():
    nim_sum ^= pile

if nim_sum == 0:
    print("Bruno")
else:
    print("Anna")