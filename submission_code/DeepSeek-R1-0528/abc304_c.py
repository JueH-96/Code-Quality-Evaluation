import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    n, D = map(int, data[0].split())
    points = []
    for i in range(1, n+1):
        x, y = map(int, data[i].split())
        points.append((x, y))
        
    D_sq = D * D
    infected = [False] * n
    infected[0] = True
    q = deque([0])
    
    while q:
        i = q.popleft()
        for j in range(n):
            if not infected[j]:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq <= D_sq:
                    infected[j] = True
                    q.append(j)
                    
    for i in range(n):
        print("Yes" if infected[i] else "No")

if __name__ == "__main__":
    main()