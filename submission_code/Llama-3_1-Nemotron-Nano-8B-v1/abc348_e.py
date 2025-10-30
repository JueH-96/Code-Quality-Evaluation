import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    c = list(map(int, sys.stdin.readline().split()))
    
    parent = [0] * (n + 1)
    S = [0] * (n + 1)
    stack = [(1, None, False)]
    while stack:
        u, p, visited = stack.pop()
        if not visited:
            stack.append((u, p, True))
            for v in adj[u]:
                if v != p:
                    parent[v] = u
                    stack.append((v, u, False))
        else:
            S[u] = c[u - 1]
            for v in adj[u]:
                if v != p:
                    S[u] += S[v]
    
    sum_root = 0
    distance = [0] * (n + 1)
    visited = [False] * (n + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        sum_root += c[u - 1] * distance[u]
        for v in adj[u]:
            if not visited[v] and v != parent[u]:
                distance[v] = distance[u] + 1
                visited[v] = True
                q.append(v)
    
    total_C = sum(c)
    min_sum = sum_root
    q = deque()
    q.append((1, sum_root, -1))
    while q:
        u, current_sum, p = q.popleft()
        for v in adj[u]:
            if v == p:
                continue
            if parent[u] == v:
                current_S_v = total_C - S[u]
            else:
                current_S_v = S[v]
            sum_v = current_sum + (total_C - 2 * current_S_v)
            if sum_v < min_sum:
                min_sum = sum_v
            q.append((v, sum_v, u))
    
    print(min_sum)

if __name__ == "__main__":
    main()