import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    parents_list = [int(next(it)) for _ in range(n-1)]
    
    graph = [[] for _ in range(n+1)]
    for idx in range(2, n+1):
        p = parents_list[idx-2]
        graph[p].append(idx)
        
    depth = [0] * (n+1)
    q = deque([1])
    while q:
        u = q.popleft()
        for v in graph[u]:
            depth[v] = depth[u] + 1
            q.append(v)
            
    insurances = [[] for _ in range(n+1)]
    for _ in range(m):
        x = int(next(it))
        y = int(next(it))
        insurances[x].append(y)
        
    best = [-10**15] * (n+1)
    q = deque([1])
    if insurances[1]:
        best[1] = max(depth[1] + y for y in insurances[1])
    else:
        best[1] = -10**15
        
    while q:
        u = q.popleft()
        for v in graph[u]:
            base = best[u]
            if insurances[v]:
                candidate = max(depth[v] + y for y in insurances[v])
                best_val = max(base, candidate)
            else:
                best_val = base
            best[v] = best_val
            q.append(v)
            
    count_covered = 0
    for i in range(1, n+1):
        if best[i] >= depth[i]:
            count_covered += 1
            
    print(count_covered)

if __name__ == "__main__":
    main()