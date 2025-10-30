import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it)); m = int(next(it)); K_val = int(next(it))
    edges = []
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        u = int(next(it)); v = int(next(it))
        edges.append((u, v))
        graph[u].append(v)
    
    reachable = set()
    queue = deque([1])
    reachable.add(1)
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in reachable:
                reachable.add(v)
                queue.append(v)
    
    in_edges = [[] for _ in range(n+1)]
    for u, v in edges:
        if v in reachable and u in reachable:
            in_edges[v].append(u)
    
    INF = 10**9
    dp = [[INF] * (K_val+1) for _ in range(n+1)]
    for c in range(0, K_val+1):
        if c == 0:
            dp[1][0] = 0
        else:
            dp[1][c] = INF
            
    changed = True
    while changed:
        changed = False
        for v in range(2, n+1):
            if v not in reachable:
                continue
            for c in range(0, K_val+1):
                in_list = in_edges[v]
                if not in_list and v != 1:
                    if c == 0:
                        new_val = 0
                    else:
                        new_val = INF
                    if new_val < dp[v][c]:
                        dp[v][c] = new_val
                        changed = True
                    continue
                
                best = INF
                for X in range(0, K_val+1):
                    total_a = 0
                    valid = True
                    for u in in_list:
                        options = []
                        if c >= 0:
                            if c <= K_val:
                                if dp[u][c] <= X:
                                    options.append(0)
                        else:
                            if dp[u][0] <= X:
                                options.append(0)
                        if c - 1 < 0:
                            options.append(1)
                        else:
                            if c - 1 <= K_val:
                                if dp[u][c-1] <= X:
                                    options.append(1)
                        if not options:
                            valid = False
                            break
                        total_a += min(options)
                    if not valid:
                        continue
                    total_cost = total_a + X
                    if total_cost < best:
                        best = total_cost
                if best < dp[v][c]:
                    dp[v][c] = best
                    changed = True
                    
    ans = 0
    for d in range(0, K_val+1):
        if dp[n][d] <= K_val:
            ans = d
    print(ans)

if __name__ == "__main__":
    main()