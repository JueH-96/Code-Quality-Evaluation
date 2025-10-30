MOD = 998244353

def sum_of_divisors(n):
    total = 1 + n
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total

def count_good_integers(N):
    count = 0
    for i in range(1, N + 1):
        if sum_of_divisors(i) % 3 == 0:
            count += 1
    return count

def power_mod(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def solve(N, M):
    good_integers_count = count_good_integers(N)
    return power_mod(good_integers_count, M, MOD)

N, M = map(int, input().split())
print(solve(N, M))