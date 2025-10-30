def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))
    
    infected = [False] * n
    infected[0] = True
    
    q = [0]
    
    while q:
        curr = q.pop(0)
        for i in range(n):
            if not infected[i]:
                dist_sq = (points[curr][0] - points[i][0])**2 + (points[curr][1] - points[i][1])**2
                if dist_sq <= d**2:
                    infected[i] = True
                    q.append(i)
                    
    for status in infected:
        print("Yes" if status else "No")

solve()