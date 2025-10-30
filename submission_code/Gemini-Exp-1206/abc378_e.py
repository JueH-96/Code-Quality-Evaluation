def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += a[j]
            total_sum += current_sum % m
            
    print(total_sum)

solve()