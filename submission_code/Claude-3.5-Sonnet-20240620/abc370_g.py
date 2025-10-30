# YOUR CODE HERE
MOD = 998244353

def sum_of_divisors(n):
    s = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            s += i
            if i * i != n:
                s += n // i
        i += 1
    return s

def is_good(n):
    return sum_of_divisors(n) % 3 == 0

def pow_mod(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def solve(N, M):
    good_count = sum(1 for i in range(1, N+1) if is_good(i))
    return pow_mod(good_count, M, MOD)

N, M = map(int, input().split())
print(solve(N, M))