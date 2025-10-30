MOD = 998244353

def power(x, y):
    res = 1
    x = x % MOD
    while y > 0:
        if y & 1:
            res = (res * x) % MOD
        y = y >> 1
        x = (x * x) % MOD
    return res

def main():
    N = int(input())
    digit = 0
    n = N
    while n > 0:
        digit += 1
        n //= 10
    res = 0
    for i in range(digit):
        res = (res + (N % 10) * power(10, i) * power(10, N * (digit - i - 1))) % MOD
        N //= 10
    print(res)

main()