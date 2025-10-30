def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_sum = 0
    mod = 998244353
    
    for l in range(n):
        for r in range(l, n):
            current_sum = 0
            for i in range(l, r + 1):
                current_sum = (current_sum + a[i]) % mod
            
            total_sum = (total_sum + pow(current_sum, k, mod)) % mod
            
    print(total_sum)

solve()