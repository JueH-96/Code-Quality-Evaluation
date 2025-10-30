def solve():
    n, m = map(int, input().split())
    
    mod = 998244353
    
    ans = (n * m * (mod + 1 - pow(2, mod - 2, mod))) % mod
    ans = (ans * (n * m + 1)) % mod
    
    print(ans)

solve()