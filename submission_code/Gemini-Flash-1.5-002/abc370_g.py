# YOUR CODE HERE
MOD = 998244353

def sum_divisors(n):
    total = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            total += i
            if i * i != n:
                total += n // i
        i += 1
    return total

def is_good(n):
    return sum_divisors(n) % 3 == 0

def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

n, m = map(int, input().split())

count = 0
dp = {}

def solve(idx, current_prod):
    if idx == m:
        if current_prod <= n and is_good(current_prod):
            return 1
        else:
            return 0
    if (idx, current_prod) in dp:
        return dp[(idx, current_prod)]
    ans = 0
    for i in range(1, int(n**0.5) + 2):
        next_prod = current_prod * i
        if next_prod > n:
            break
        ans = (ans + solve(idx + 1, next_prod)) % MOD
    dp[(idx, current_prod)] = ans
    return ans

print(solve(0, 1))