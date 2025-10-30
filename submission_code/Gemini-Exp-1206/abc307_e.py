def solve():
    n, m = map(int, input().split())
    mod = 998244353
    
    if n == 2:
        print((m * (m - 1)) % mod)
        return
    
    if m == 1:
        print(0)
        return

    ans = (m * pow(m - 1, n - 1, mod)) % mod
    
    if n % 2 == 0:
        ans = (ans - (m * pow(m-1,n//2 -1,mod) * pow(m-2,n//2,mod)) % mod + mod) % mod
        ans = (ans + (m * pow(m-1,n//2 -1,mod)) % mod) % mod
    else:
        ans = (ans - (m * pow(m-1,(n-1)//2 -1,mod) * pow(m-2,(n-1)//2 + 1,mod)) % mod + mod) % mod
        ans = (ans + (m * pow(m-1,(n-1)//2 -1,mod) * (m-1)) % mod) % mod
    
    print(ans)

solve()