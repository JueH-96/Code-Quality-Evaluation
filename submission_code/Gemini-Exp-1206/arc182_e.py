def solve():
    n, m, c, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for k_val in range(k):
        min_val = float('inf')
        for i in range(n):
            min_val = min(min_val, (c * k_val + a[i]) % m)
        total_sum += min_val
    
    print(total_sum)

solve()