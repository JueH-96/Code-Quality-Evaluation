def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    result = []
    for i in range(n - 1):
        result.append(a[i] * a[i+1])
    
    print(*result)

solve()