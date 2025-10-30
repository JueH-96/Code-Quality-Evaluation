from collections import deque

def can_reach(x1, y1, x2, y2):
    # If sum of differences is odd, it's impossible to reach
    if (abs(x2-x1) + abs(y2-y1)) % 2 != 0:
        return 0
        
    # If points are same, distance is 0
    if x1 == x2 and y1 == y2:
        return 0
        
    # If points are too far apart, impossible to reach
    if abs(x2-x1) > abs(y2-y1) + 20 or abs(y2-y1) > abs(x2-x1) + 20:
        return 0
        
    # BFS to find shortest path
    visited = set()
    q = deque([(x1, y1, 0)])
    visited.add((x1, y1))
    
    while q:
        x, y, d = q.popleft()
        
        if x == x2 and y == y2:
            return d
            
        for nx, ny in [(x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)]:
            if (nx,ny) not in visited:
                visited.add((nx,ny))
                q.append((nx,ny,d+1))
                
    return 0

N = int(input())
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

ans = 0
for i in range(N-1):
    for j in range(i+1, N):
        x1, y1 = points[i]
        x2, y2 = points[j]
        ans += can_reach(x1, y1, x2, y2)
        
print(ans)