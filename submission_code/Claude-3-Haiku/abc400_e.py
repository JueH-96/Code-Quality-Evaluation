import math

def is_400_number(n):
    prime_factors = set()
    while n % 2 == 0:
        prime_factors.add(2)
        n //= 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            prime_factors.add(i)
            n //= i
    if n > 2:
        prime_factors.add(n)
    return len(prime_factors) == 2 and all(n % p ** 2 == 0 for p in prime_factors)

def solve_query(a):
    for n in range(a, 35, -1):
        if is_400_number(n):
            return n
    return 36

q = int(input())
for _ in range(q):
    a = int(input())
    print(solve_query(a))