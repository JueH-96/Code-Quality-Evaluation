def solve():
    n = int(input())
    mod = 998244353
    
    s_n = str(n)
    len_n = len(s_n)
    
    ans = 0
    power_of_10 = 1
    
    for _ in range(n):
        ans = (ans + n * power_of_10) % mod
        power_of_10 = (power_of_10 * (10 ** len_n)) % mod
    
    print(ans)

solve()