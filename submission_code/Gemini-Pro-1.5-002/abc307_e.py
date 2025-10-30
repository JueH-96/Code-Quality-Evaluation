# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    mod = 998244353

    if n == 2:
        print(m * (m - 1) % mod)
        return

    ans = pow(m - 2, n - 1, mod) * m % mod
    ans = (ans + pow(-1, n, mod) * (m - 2) * pow(m-1, n-1, mod)) % mod
    
    if ans < 0:
        ans += mod
    
    print(ans % mod)

solve()