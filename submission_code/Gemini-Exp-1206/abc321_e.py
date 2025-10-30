def solve():
    n, x, k = map(int, input().split())
    
    ans = 0
    q = [(x, 0)]
    visited = set()
    
    while q:
        curr, dist = q.pop(0)
        
        if (curr, dist) in visited:
            continue
        visited.add((curr, dist))
        
        if dist == k:
            ans += 1
        
        if dist < k:
            if curr * 2 <= n:
                q.append((curr * 2, dist + 1))
            if curr * 2 + 1 <= n:
                q.append((curr * 2 + 1, dist + 1))
            if curr // 2 >= 1:
                q.append((curr // 2, dist + 1))
    
    print(ans)

t = int(input())
for _ in range(t):
    solve()