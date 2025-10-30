def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    ans = 0
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += a[j]
            ans += current_sum % m
            
    print(ans)

solve()