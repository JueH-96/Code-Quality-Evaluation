def solve():
    n, m, c, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_min = 0
    for i in range(k):
        min_val = float('inf')
        for j in range(n):
            val = (c * i + a[j]) % m
            min_val = min(min_val, val)
        total_min += min_val
    
    print(total_min)

solve()