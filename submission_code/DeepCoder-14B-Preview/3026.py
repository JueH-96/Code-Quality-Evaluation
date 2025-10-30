import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    if N == 0:
        return
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    for _ in range(N - 1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
        degrees[a] += 1
        degrees[b] += 1

    removed = [False] * (N + 1)
    result = []

    while True:
        leaves = [i for i in range(1, N + 1) if not removed[i] and degrees[i] == 1]
        if len(leaves) < 2:
            break

        max_dist = -1
        best_pair = None

        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                u = leaves[i]
                v = leaves[j]
                visited = [False] * (N + 1)
                q = deque()
                q.append((u, 0))
                visited[u] = True
                found = False
                while q:
                    node, d = q.popleft()
                    if node == v:
                        if d > max_dist:
                            max_dist = d
                            best_pair = (u, v)
                        found = True
                        break
                    for neighbor in adj[node]:
                        if not visited[neighbor] and not removed[neighbor]:
                            visited[neighbor] = True
                            q.append((neighbor, d + 1))
                if found:
                    break
            if best_pair:
                break

        if best_pair is None:
            break

        u, v = best_pair
        result.append((u, v))
        removed[u] = True
        removed[v] = True

        for node in [u, v]:
            for neighbor in adj[node]:
                if not removed[neighbor]:
                    degrees[neighbor] -= 1
            adj[node] = []

    for pair in result:
        print(pair[0], pair[1])

if __name__ == '__main__':
    main()