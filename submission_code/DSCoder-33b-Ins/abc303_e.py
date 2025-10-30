from collections import defaultdict

def solve():
    n = int(input())
    adj = defaultdict(list)
    degree = defaultdict(int)
    for _ in range(n-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1

    leaves = [node for node in adj if len(adj[node]) == 1]
    levels = []
    while leaves:
        leaf = leaves.pop()
        if degree[leaf] == 0:
            continue
        degree[leaf] -= 1
        degree[adj[leaf][0]] -= 1
        adj[adj[leaf][0]].remove(leaf)
        adj[leaf].remove(adj[leaf][0])
        if len(adj[adj[leaf][0]]) == 1:
            leaves.append(adj[leaf][0])
        levels.append(len(adj[leaf]))

    levels.sort()
    print(*levels)

solve()