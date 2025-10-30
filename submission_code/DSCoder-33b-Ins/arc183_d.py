# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    edges = [list(map(int, input().split())) for _ in range(n-1)]
    graph = [[] for _ in range(n+1)]
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    leaves = [i for i in range(1, n+1) if len(graph[i]) == 1]
    while leaves:
        leaf = leaves.pop()
        if not graph[leaf]:
            continue
        parent = graph[leaf][0]
        print(leaf, parent)
        graph[leaf] = []
        graph[parent].remove(leaf)
        if len(graph[parent]) == 1:
            leaves.append(parent)

solve()