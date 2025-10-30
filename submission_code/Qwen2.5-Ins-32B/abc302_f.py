import sys
from collections import defaultdict, deque

def solve():
    N, M = map(int, input().split())
    sets = [set(map(int, input().split())) for _ in range(N)]
    graph = defaultdict(set)
    for i, s in enumerate(sets):
        for j, t in enumerate(sets):
            if i != j and s & t:
                graph[i].add(j)
                graph[j].add(i)
    visited = [False] * N
    queue = deque([i for i, s in enumerate(sets) if 1 in s])
    steps = 0
    while queue:
        for _ in range(len(queue)):
            i = queue.popleft()
            if M in sets[i]:
                return steps
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    queue.append(j)
                    sets[j] |= sets[i]
        steps += 1
    return -1

print(solve())