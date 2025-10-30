import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    graph = [[] for _ in range(n)]
    total = 0
    index = 1
    for _ in range(n-1):
        a = int(data[index])
        b = int(data[index+1])
        c = int(data[index+2])
        index += 3
        a -= 1
        b -= 1
        graph[a].append((b, c))
        graph[b].append((a, c))
        total += c
    
    if n == 1:
        print(0)
        return
        
    dist0 = [-1] * n
    q = deque()
    dist0[0] = 0
    q.append(0)
    while q:
        cur = q.popleft()
        for neighbor, w in graph[cur]:
            if dist0[neighbor] == -1:
                dist0[neighbor] = dist0[cur] + w
                q.append(neighbor)
    
    u = 0
    for i in range(1, n):
        if dist0[i] > dist0[u]:
            u = i
            
    distu = [-1] * n
    q = deque()
    distu[u] = 0
    q.append(u)
    while q:
        cur = q.popleft()
        for neighbor, w in graph[cur]:
            if distu[neighbor] == -1:
                distu[neighbor] = distu[cur] + w
                q.append(neighbor)
                
    diameter = max(distu)
    ans = 2 * total - diameter
    print(ans)

if __name__ == '__main__':
    main()