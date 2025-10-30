def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    def power(base, exp):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return res
        
    def inverse(n):
        return power(n, mod - 2)
        
    if n == 0:
        print(0)
        return
        
    inv_n = inverse(n)
    r = ((n + 1) * inv_n) % mod
    
    s = 0
    for j in range(1, n + 1):
        term = (a[j-1] * power(r, j - 1)) % mod
        s = (s + term) % mod
        
    expected_value = (inv_n * s) % mod
    print(expected_value)

if __name__ == '__main__':
    solve()