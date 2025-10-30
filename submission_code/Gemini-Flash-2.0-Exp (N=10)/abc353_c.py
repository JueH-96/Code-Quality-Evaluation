def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = 0
    mod = 10**8
    
    for i in range(n):
        for j in range(i + 1, n):
            total_sum += (a[i] + a[j]) % mod
            
    print(total_sum)

solve()