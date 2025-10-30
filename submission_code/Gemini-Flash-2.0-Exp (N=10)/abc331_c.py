def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    results = []
    for i in range(n):
        current_sum = 0
        for j in range(n):
            if a[j] > a[i]:
                current_sum += a[j]
        results.append(current_sum)
    
    print(*results)

solve()