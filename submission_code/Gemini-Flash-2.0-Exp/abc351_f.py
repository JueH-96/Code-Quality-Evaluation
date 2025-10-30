def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sum += max(a[j] - a[i], 0)
    
    print(total_sum)

solve()