# YOUR CODE HERE
import itertools

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    min_weight = float('inf')
    
    for perm in itertools.permutations(range(1, n + 1)):
        total_weight = 0
        valid_path = True
        for i in range(len(perm) - 1):
            u = perm[i]
            v = perm[i+1]
            found_edge = False
            for edge in edges:
                if edge[0] == u and edge[1] == v:
                    total_weight += edge[2]
                    found_edge = True
                    break
            if not found_edge:
                valid_path = False
                break
        if valid_path:
            min_weight = min(min_weight, total_weight)

    if min_weight == float('inf'):
        print("No")
    else:
        print(min_weight)

solve()