from collections import defaultdict

def solve():
    N = int(input())
    edges = []
    for i in range(N-1):
        a, b = map(int, input().split())
        edges.append((a, b))

    # Build the adjacency list
    adj_list = defaultdict(list)
    for a, b in edges:
        adj_list[a].append(b)
        adj_list[b].append(a)

    # Find the leaves
    leaves = [i for i in range(1, N+1) if len(adj_list[i]) == 1]

    # Perform the operations
    result = []
    while len(leaves) >= 2:
        # Choose the two leaves with the maximum distance
        max_distance = 0
        max_pair = None
        for i in range(0, len(leaves), 2):
            a, b = leaves[i], leaves[i+1]
            distance = 0
            u, v = a, b
            while u != v:
                if u < v:
                    u = adj_list[u][0]
                else:
                    v = adj_list[v][0]
                distance += 1
            if distance > max_distance:
                max_distance = distance
                max_pair = (leaves[i], leaves[i+1])

        # Remove the chosen leaves
        result.append(max_pair)
        for leaf in max_pair:
            leaves.remove(leaf)
            adj_list[adj_list[leaf][0]].remove(leaf)

    # Print the result
    for x, y in result:
        print(x, y)

solve()