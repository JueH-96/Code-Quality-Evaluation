import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    s = int(data[2]) - 1
    t = int(data[3]) - 1
    graph = [[] for _ in range(n)]
    index = 4
    for _ in range(m):
        u = int(data[index]) - 1
        v = int(data[index+1]) - 1
        index += 2
        graph[u].append(v)
        graph[v].append(u)
    
    da = [-1] * n
    q = deque()
    da[s] = 0
    q.append(s)
    while q:
        u = q.popleft()
        for v in graph[u]:
            if da[v] == -1:
                da[v] = da[u] + 1
                q.append(v)
                
    db = [-1] * n
    q.append(t)
    db[t] = 0
    while q:
        u = q.popleft()
        for v in graph[u]:
            if db[v] == -1:
                db[v] = db[u] + 1
                q.append(v)
                
    d = da[t]
    found = False
    for u in range(n):
        for v in graph[u]:
            if da[u] == -1 or db[v] == -1 or da[v] == -1 or db[u] == -1:
                continue
            if da[u] + 1 + db[v] >= d + 1 and da[v] + 1 + db[u] >= d + 1:
                found = True
                break
        if found:
            break
            
    print(d + 2 if found else -1)

if __name__ == '__main__':
    main()