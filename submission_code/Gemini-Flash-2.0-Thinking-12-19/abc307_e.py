def power(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def solve():
    n, m = map(int, input().split())
    if m < 2:
        print(0)
        return
    mod = 998244353
    m_minus_1 = (m - 1) % mod
    term1 = power(m_minus_1, n, mod)
    term2 = m_minus_1
    if n % 2 == 0:
        result = (term1 + term2) % mod
    else:
        result = (term1 - term2) % mod
        if result < 0:
            result += mod
    print(result)

if __name__ == '__main__':
    solve()