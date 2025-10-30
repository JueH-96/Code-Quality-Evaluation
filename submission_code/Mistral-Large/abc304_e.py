import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    edges = []
    for _ in range(M):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        edges.append((u, v))

    K = int(data[index])
    index += 1

    restrictions = []
    for _ in range(K):
        x = int(data[index])
        index += 1
        y = int(data[index])
        index += 1
        restrictions.append((x, y))

    Q = int(data[index])
    index += 1

    queries = []
    for _ in range(Q):
        p = int(data[index])
        index += 1
        q = int(data[index])
        index += 1
        queries.append((p, q))

    from collections import defaultdict
    from sys import setrecursionlimit
    setrecursionlimit(10**6)

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * (N + 1)
    component = [-1] * (N + 1)

    def dfs(u, comp):
        visited[u] = True
        component[u] = comp
        for v in graph[u]:
            if not visited[v]:
                dfs(v, comp)

    comp = 0
    for i in range(1, N + 1):
        if not visited[i]:
            dfs(i, comp)
            comp += 1

    restriction_set = set(restrictions)

    def is_good(p, q):
        if component[p] == component[q]:
            return False
        new_restrictions = set()
        for x, y in restriction_set:
            if (component[x] == component[p] and component[y] == component[q]) or \
               (component[x] == component[q] and component[y] == component[p]):
                new_restrictions.add((x, y))
        if new_restrictions:
            return False
        return True

    results = []
    for p, q in queries:
        if is_good(p, q):
            results.append("Yes")
        else:
            results.append("No")

    print("
".join(results))

if __name__ == "__main__":
    main()