MOD = 998244353

def power(x, n):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return (x * power(x, n - 1)) % MOD
    else:
        half = power(x, n // 2)
        return (half * half) % MOD

def solution(n):
    # Calculate 10^len(str(n)) % MOD
    base = 10 ** len(str(n))
    # The formula is (base^k - 1) * n / (base - 1), where k is the length of the concatenated string (n times)
    # First calculate (base^n - 1) % MOD
    base_n_minus_one = (power(base, n) - 1) % MOD
    # Then calculate (base_n_minus_one * n) % MOD
    temp = (base_n_minus_one * n) % MOD
    # Finally, calculate temp * power(base - 1, MOD - 2) % MOD where power(base - 1, MOD - 2) is the modular inverse of (base - 1)
    result = (temp * power(base - 1, MOD - 2)) % MOD
    return result

N = int(input())
print(solution(N))