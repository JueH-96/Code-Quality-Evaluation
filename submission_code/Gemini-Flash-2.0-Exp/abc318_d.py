def solve():
    n = int(input())
    d = []
    for i in range(n - 1):
        d.append(list(map(int, input().split())))

    edges = []
    for i in range(n - 1):
        for j in range(len(d[i])):
            edges.append((i + 1, i + 2 + j, d[i][j]))

    max_weight = 0
    for i in range(1 << len(edges)):
        chosen_edges = []
        total_weight = 0
        vertices = set()

        for j in range(len(edges)):
            if (i >> j) & 1:
                u, v, weight = edges[j]
                chosen_edges.append((u, v, weight))
                total_weight += weight
                vertices.add(u)
                vertices.add(v)

        if len(vertices) == 2 * len(chosen_edges):
            max_weight = max(max_weight, total_weight)

    print(max_weight)

solve()