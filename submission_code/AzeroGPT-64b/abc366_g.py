from collections import defaultdict

def read_ints():
    return map(int, input().split())

def solve():
    N, M = read_ints()
    graph = defaultdict(set)
    for _ in range(M):
        u, v = read_ints()
        graph[u].add(v)
        graph[v].add(u)

    # Find nodes with odd degree, there should be at most 2 such nodes
    odd_nodes = [u for u, neighbors in graph.items() if len(neighbors) % 2 == 1]
    if len(odd_nodes) > 2:
        print("No")
        return

    if len(odd_nodes) == 2:
        # Odd case: Assign half of the nodes 1 and half 2 so that the XOR of their neighbors is 0
        x, y = odd_nodes
        for u in graph:
            if u == x:
                graph[u].add(y)
            elif u == y:
                graph[u].add(x)

    print("Yes")
    print(" ".join("1" if len(graph[u]) % 2 == 1 else "2" for u in range(1, N + 1)))

solve()