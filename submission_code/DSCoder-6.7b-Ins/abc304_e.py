import sys
from collections import defaultdict

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]
    K = int(sys.stdin.readline())
    bad_pairs = [tuple(map(int, sys.stdin.readline().split())) for _ in range(K)]
    Q = int(sys.stdin.readline())
    queries = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    return N, M, edges, K, bad_pairs, Q, queries

def solve():
    N, M, edges, K, bad_pairs, Q, queries = read_input()
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    for x, y in bad_pairs:
        if x in graph[y] or y in graph[x]:
            print("No")
            return
    for p, q in queries:
        if p in graph[q] or q in graph[p]:
            print("No")
        else:
            print("Yes")

solve()