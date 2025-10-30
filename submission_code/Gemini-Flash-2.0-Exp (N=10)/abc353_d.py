def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = 0
    mod = 998244353
    
    for i in range(n):
        for j in range(i + 1, n):
            s1 = str(a[i])
            s2 = str(a[j])
            
            combined_str = s1 + s2
            combined_int = int(combined_str)
            total_sum = (total_sum + combined_int) % mod
            
    print(total_sum)

solve()