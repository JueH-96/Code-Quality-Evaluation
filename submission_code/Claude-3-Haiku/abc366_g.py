from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)

    # Check if there is a solution
    for v in range(1, N+1):
        if len(graph[v]) >= 1:
            neighbors = graph[v]
            xor_sum = 0
            for u in neighbors:
                xor_sum ^= u
            if xor_sum != 0:
                print("No")
                return

    # Find a solution
    values = [0] * (N+1)
    for v in range(1, N+1):
        if len(graph[v]) >= 1:
            neighbors = graph[v]
            xor_sum = 0
            for u in neighbors:
                xor_sum ^= values[u]
            values[v] = xor_sum

    print("Yes")
    print(" ".join(map(str, values[1:])))

solve()