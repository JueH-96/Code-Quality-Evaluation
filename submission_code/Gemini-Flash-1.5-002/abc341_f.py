# YOUR CODE HERE
n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u - 1, v - 1))
weights = list(map(int, input().split()))
pieces = list(map(int, input().split()))

graph = [[] for _ in range(n)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

total_pieces = sum(pieces)
ans = 0
while sum(pieces) > 0:
    best_vertex = -1
    max_pieces = -1
    for i in range(n):
        if pieces[i] > 0:
            if pieces[i] > max_pieces:
                max_pieces = pieces[i]
                best_vertex = i
    
    pieces[best_vertex] -= 1
    ans += 1
    
    
    best_subset = []
    min_weight_sum = float('inf')

    for i in range(1 << len(graph[best_vertex])):
        subset = []
        weight_sum = 0
        for j in range(len(graph[best_vertex])):
            if (i >> j) & 1:
                subset.append(graph[best_vertex][j])
                weight_sum += weights[graph[best_vertex][j]]
        if weight_sum < weights[best_vertex] and weight_sum < min_weight_sum:
            min_weight_sum = weight_sum
            best_subset = subset

    for v in best_subset:
        pieces[v] += 1

print(ans)