def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            total_sum += (a[i] + a[j]) % 100000000
    
    print(total_sum)

solve()