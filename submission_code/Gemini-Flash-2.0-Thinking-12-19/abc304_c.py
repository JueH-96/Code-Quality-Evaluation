import math

def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))
    
    adj = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            if dist_sq <= d**2:
                adj[i].append(j)
                adj[j].append(i)
                
    infected = [False] * n
    infected[0] = True
    
    queue = [0]
    
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not infected[v]:
                infected[v] = True
                queue.append(v)
                
    for i in range(n):
        if infected[i]:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    solve()