def solve():
    n, m = map(int, input().split())
    
    ans = 0
    for k in range(n + 1):
        ans = (ans + bin(k & m).count('1')) % 998244353
    print(ans)

solve()