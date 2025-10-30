# YOUR CODE HERE
from collections import deque
import sys

input = sys.stdin.read
sys.setrecursionlimit(10**7)

def solve():
    n, m, k = map(int, input().split())
    edges = [[] for _ in range(n)]
    for _ in range(m):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        edges[a].append(b)
        edges[b].append(a)
    guards = [0] * n
    for _ in range(k):
        p, h = map(int, input().split())
        p -= 1
        guards[p] = h
    visited = [False] * n
    queue = deque()
    for i in range(n):
        if guards[i] > 0:
            queue.append(i)
            visited[i] = True
    while queue:
        current = queue.popleft()
        for neighbor in edges[current]:
            if guards[neighbor] <= guards[current] - 1 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    guarded_vertices = [i + 1 for i, v in enumerate(visited) if v]
    print(len(guarded_vertices))
    print(' '.join(map(str, guarded_vertices)))

solve()