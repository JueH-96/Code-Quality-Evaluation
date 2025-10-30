def solve():
    n, m = map(int, input().split())
    photos = []
    for _ in range(m):
        photos.append(list(map(int, input().split())))
    
    adjacent = [[False] * (n + 1) for _ in range(n + 1)]
    
    for photo in photos:
        for i in range(n - 1):
            adjacent[photo[i]][photo[i+1]] = True
            adjacent[photo[i+1]][photo[i]] = True
    
    count = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if not adjacent[i][j]:
                count += 1
    
    print(count)

solve()