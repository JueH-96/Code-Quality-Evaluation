import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(N-1)]
    G = [[] for _ in range(N+1)]
    for a, b in edges:
        G[a].append(b)
        G[b].append(a)

    # Find the longest path in the tree
    dist = [-1] * (N+1)
    dist[0] = 0
    dist[1] = 0
    que = deque([1])
    while que:
        v = que.popleft()
        for w in G[v]:
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                que.append(w)
    v = dist.index(max(dist))
    dist = [-1] * (N+1)
    dist[v] = 0
    que = deque([v])
    while que:
        v = que.popleft()
        for w in G[v]:
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                que.append(w)
    u = dist.index(max(dist))

    # Find the middle point of the longest path
    path = [u]
    while v != u:
        for w in G[v]:
            if dist[w] == dist[v] - 1:
                path.append(w)
                v = w
                break
    mid = path[len(path)//2]

    # Remove leaves from the tree
    removed = [False] * (N+1)
    removed[0] = True
    que = deque([mid])
    while que:
        v = que.popleft()
        for w in G[v]:
            if not removed[w]:
                removed[w] = True
                que.append(w)
    for v in range(1, N+1):
        if not removed[v]:
            print(v, end=' ')
    print()

if __name__ == "__main__":
    main()