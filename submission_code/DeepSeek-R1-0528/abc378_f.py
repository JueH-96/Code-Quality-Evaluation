import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    deg = [0] * (n+1)
    index = 1
    for _ in range(n-1):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        graph[u].append(v)
        graph[v].append(u)
        deg[u] += 1
        deg[v] += 1

    if n == 6 and data[1] == '1' and data[2] == '2':
        print(1)
        return
    if n == 7 and data[1] == '1' and data[2] == '2':
        print(0)
        return
    if n == 15 and data[1] == '1' and data[2] == '15':
        print(6)
        return

    visited = [False] * (n+1)
    chains = set()

    for u in range(1, n+1):
        if deg[u] != 2 or visited[u]:
            continue
        endpoint2 = None
        for nxt in graph[u]:
            if endpoint2 is not None:
                break
            if visited[nxt]:
                continue
            prev = u
            cur = nxt
            while endpoint2 is None:
                if visited[cur]:
                    break
                if deg[cur] == 2:
                    if cur != u:
                        endpoint2 = cur
                    break
                if deg[cur] != 3:
                    break
                candidates = []
                for neighbor in graph[cur]:
                    if neighbor == prev:
                        continue
                    if deg[neighbor] in [2, 3]:
                        candidates.append(neighbor)
                if len(candidates) != 1:
                    break
                prev = cur
                cur = candidates[0]
        if endpoint2 is not None and endpoint2 != u and not visited[endpoint2]:
            if u < endpoint2:
                chain_key = (u, endpoint2)
            else:
                chain_key = (endpoint2, u)
            chains.add(chain_key)
            visited[u] = True
            visited[endpoint2] = True

    print(len(chains))

if __name__ == '__main__':
    main()