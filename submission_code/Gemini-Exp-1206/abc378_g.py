def solve():
    a, b, m = map(int, input().split())
    n = a * b - 1
    
    if a == 1 or b == 1:
        print(0)
        return

    fact = [1] * (n + 1)
    inv = [1] * (n + 1)
    factinv = [1] * (n + 1)

    for i in range(2, n + 1):
        fact[i] = fact[i - 1] * i % m
        inv[i] = -(m // i) * inv[m % i] % m
        factinv[i] = factinv[i - 1] * inv[i] % m

    def cmb(n, r):
        if r < 0 or r > n:
            return 0
        return fact[n] * factinv[r] % m * factinv[n - r] % m

    def f(x, y):
        return cmb(x + y - 2, x - 1)

    ans = (f(a, b) * f(a, b) - f(a - 1, b + 1) * f(a + 1, b - 1)) % m
    
    print(ans)

solve()