def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    ans = 0
    for i in range(n):
        for j in range(i, n):
            ans += len(set(a[i:j+1]))
    
    print(ans)

solve()