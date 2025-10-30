def power(base, exp, modulus):
    res = 1
    base %= modulus
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % modulus
        exp >>= 1
        base = (base * base) % modulus
    return res

def solve():
    n, m = map(int, input().split())
    if m == 1:
        if n >= 2:
            print(0)
        else:
            print(1)
        return
    
    modulus = 998244353
    x = m - 1
    term1 = power(x, n, modulus)
    term2 = 0
    if n % 2 == 0:
        term2 = x
    else:
        term2 = -x
    
    result = (term1 + term2) % modulus
    print(result)

if __name__ == '__main__':
    solve()