def solve():
    n, m = map(int, input().split())
    mod = 998244353
    ans = 0
    
    for k in range(n + 1):
        ans = (ans + bin(k & m).count('1')) % mod
    
    print(ans)

solve()