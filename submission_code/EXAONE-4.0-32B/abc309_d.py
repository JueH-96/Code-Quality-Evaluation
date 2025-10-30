import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n1 = int(next(it))
    n2 = int(next(it))
    m = int(next(it))
    n = n1 + n2
    graph = [[] for _ in range(n+1)]
    
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        graph[a].append(b)
        graph[b].append(a)
    
    distA = [-1] * (n+1)
    q = deque([1])
    distA[1] = 0
    while q:
        u = q.popleft()
        for v in graph[u]:
            if distA[v] == -1:
                distA[v] = distA[u] + 1
                q.append(v)
    
    distB = [-1] * (n+1)
    q = deque([n])
    distB[n] = 0
    while q:
        u = q.popleft()
        for v in graph[u]:
            if distB[v] == -1:
                distB[v] = distB[u] + 1
                q.append(v)
    
    maxA = 0
    for i in range(1, n1+1):
        if distA[i] > maxA:
            maxA = distA[i]
    
    maxB = 0
    for i in range(n1+1, n+1):
        if distB[i] > maxB:
            maxB = distB[i]
    
    ans = 1 + maxA + maxB
    print(ans)

if __name__ == "__main__":
    main()