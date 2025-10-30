import sys
from sys import stdin
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(stdin.readline())
    adj = [[] for _ in range(N+1)]
    degree = [0] * (N+1)
    for _ in range(N-1):
        u, v = map(int, stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1

    visited = [False] * (N+1)
    ans = 0

    for u in range(1, N+1):
        if degree[u] != 2 or visited[u]:
            continue
        for v in adj[u]:
            if degree[v] == 3 and not visited[v]:
                current = v
                prev = u
                path = [u, current]
                valid = True
                while True:
                    if degree[current] == 2:
                        if all(degree[node] == 3 for node in path[1:-1]):
                            ans += 1
                            visited[u] = True
                            visited[current] = True
                        break
                    elif degree[current] == 3:
                        next_nodes = [n for n in adj[current] if n != prev]
                        if len(next_nodes) != 1:
                            valid = False
                            break
                        prev = current
                        current = next_nodes[0]
                        path.append(current)
                    else:
                        valid = False
                        break
                if valid and degree[current] == 2:
                    break
        visited[u] = True

    print(ans)

main()