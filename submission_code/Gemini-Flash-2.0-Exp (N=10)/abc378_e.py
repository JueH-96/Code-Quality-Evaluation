def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for l in range(n):
        for r in range(l, n):
            current_sum = 0
            for i in range(l, r + 1):
                current_sum += a[i]
            total_sum += (current_sum % m)
    
    print(total_sum)

solve()