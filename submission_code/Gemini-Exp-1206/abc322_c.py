def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    for i in range(1, n + 1):
        ans = float('inf')
        for j in range(m):
            if a[j] >= i:
                ans = a[j] - i
                break
        print(ans)

solve()