def solve():
    n, m = map(int, input().split())
    photos = []
    for _ in range(m):
        photos.append(list(map(int, input().split())))
    
    adj = [[False] * (n + 1) for _ in range(n + 1)]
    
    for photo in photos:
        for i in range(n - 1):
            p1 = photo[i]
            p2 = photo[i+1]
            adj[p1][p2] = True
            adj[p2][p1] = True
            
    count = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if not adj[i][j]:
                count += 1
    print(count)

solve()