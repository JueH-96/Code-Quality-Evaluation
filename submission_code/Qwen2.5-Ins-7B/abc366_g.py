# YOUR CODE HERE
from collections import defaultdict

def solve(N, M, edges):
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    if N == 1:
        print("Yes")
        print(1)
        return
    
    for v in range(1, N+1):
        if len(graph[v]) == 0:
            continue
        degree = len(graph[v])
        if degree == 1:
            print("No")
            return
        if degree == 2:
            u, w = graph[v]
            if u in graph[w]:
                print("No")
                return
            print("Yes")
            x = 1
            y = 1
            z = x ^ y
            for i in range(1, N+1):
                if i in graph[v]:
                    print(x if i == u else y)
                else:
                    print(z)
            return
    
    print("Yes")
    x = 1
    for i in range(1, N+1):
        print(x)
        x = x ^ 1

N, M = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(M)]
solve(N, M, edges)