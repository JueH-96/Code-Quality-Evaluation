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

def calculate_grundy(n):
    if n == 0:
        return 0
    
    factors = prime_factorization(n)
    grundy = 0
    for count in factors.values():
        grundy += count
    return grundy

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    grundy_sum = 0
    for x in a:
        grundy_sum ^= calculate_grundy(x)
    
    if grundy_sum == 0:
        print("Bruno")
    else:
        print("Anna")

solve()