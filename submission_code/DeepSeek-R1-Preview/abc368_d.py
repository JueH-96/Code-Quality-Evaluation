import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    N, K = map(int, stdin.readline().split())
    edges = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a, b = map(int, stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)
    V = list(map(int, stdin.readline().split()))
    in_S = set(V)
    total_S = K

    root = 1
    cnt = [0] * (N + 1)
    stack = [(root, None, False)]
    
    while stack:
        u, parent, visited = stack.pop()
        if not visited:
            stack.append((u, parent, True))
            for v in edges[u]:
                if v != parent:
                    stack.append((v, u, False))
        else:
            if u in in_S:
                cnt[u] = 1
            else:
                cnt[u] = 0
            for v in edges[u]:
                if v != parent:
                    cnt[u] += cnt[v]
    
    edge_count = 0
    for u in range(2, N + 1):
        if cnt[u] >= 1 and (total_S - cnt[u]) >= 1:
            edge_count += 1
    print(edge_count + 1)

if __name__ == '__main__':
    main()