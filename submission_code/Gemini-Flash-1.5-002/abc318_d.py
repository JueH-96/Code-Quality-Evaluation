# YOUR CODE HERE
n = int(input())
d = []
for i in range(n - 1):
    d.append(list(map(int, input().split())))

edges = []
for i in range(n - 1):
    for j in range(i + 1, n):
        edges.append((i + 1, j + 1, d[i][j - i - 1]))

max_weight = 0
for i in range(1 << len(edges)):
    chosen_edges = []
    total_weight = 0
    vertices = set()
    for j in range(len(edges)):
        if (i >> j) & 1:
            chosen_edges.append(edges[j])
            vertices.add(edges[j][0])
            vertices.add(edges[j][1])
    
    valid = True
    if len(vertices) != 2 * len(chosen_edges):
        valid = False

    if valid:
        for j in range(len(chosen_edges)):
            total_weight += chosen_edges[j][2]
        max_weight = max(max_weight, total_weight)

print(max_weight)