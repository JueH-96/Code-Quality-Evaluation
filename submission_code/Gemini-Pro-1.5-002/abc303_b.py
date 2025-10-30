# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = []
    for _ in range(m):
        a.append(list(map(int, input().split())))

    adj = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(m):
        for j in range(n - 1):
            adj[a[i][j]][a[i][j+1]] = True
            adj[a[i][j+1]][a[i][j]] = True
    
    ans = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if not adj[i][j]:
                ans += 1
    print(ans)

solve()