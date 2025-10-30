from collections import deque
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    grid = []
    for i in range(1, 1+n):
        grid.append(data[i].strip())
    
    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            c = grid[i][j]
            if c != '-':
                outgoing[i].append((j, c))
                incoming[j].append((i, c))
                
    dp = [[-1] * n for _ in range(n)]
    q = deque()
    
    for i in range(n):
        dp[i][i] = 0
        q.append((i, i))
        
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if grid[i][j] != '-':
                if dp[i][j] == -1:
                    dp[i][j] = 1
                    q.append((i, j))
                    
    while q:
        u, v = q.popleft()
        for x, c1 in incoming[u]:
            for y, c2 in outgoing[v]:
                if c1 == c2:
                    if dp[x][y] == -1:
                        dp[x][y] = dp[u][v] + 2
                        q.append((x, y))
                        
    for i in range(n):
        line = []
        for j in range(n):
            line.append(str(dp[i][j]))
        print(" ".join(line))

if __name__ == "__main__":
    main()