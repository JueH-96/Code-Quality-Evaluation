import sys
from heapq import *

def solve():
    n = int(sys.stdin.readline().strip())
    edges = [0] * (n + 1)
    graph = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a, b = map(int, sys.stdin.readline().strip().split())
        graph[a].append(b)
        graph[b].append(a)
        edges[a] += 1
        edges[b] += 1

    leaves = [i for i in range(1, n + 1) if edges[i] == 1]
    heapify(leaves)

    removed = [False] * (n + 1)
    result = []
    while len(leaves) > 1:
        a = heappop(leaves)
        b = heappop(leaves)
        result.append((a, b))
        for c in graph[a]:
            if not removed[c]:
                edges[c] -= 1
                if edges[c] == 1:
                    heappush(leaves, c)
        for c in graph[b]:
            if not removed[c]:
                edges[c] -= 1
                if edges[c] == 1:
                    heappush(leaves, c)
        removed[a] = True
        removed[b] = True

    return result


if __name__ == "__main__":
    result = solve()
    for a, b in result:
        print(a, b)