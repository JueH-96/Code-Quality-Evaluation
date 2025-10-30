import sys
import itertools

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    K = int(data[2])

    edges = []
    for i in range(M):
        u = int(data[3 + 3 * i])
        v = int(data[4 + 3 * i])
        w = int(data[5 + 3 * i])
        edges.append((w, u, v))

    min_cost = float('inf')

    # Generate all possible spanning trees using combinations of edges
    for subset in itertools.combinations(edges, N - 1):
        # Check if the subset forms a spanning tree
        parent = list(range(N + 1))
        rank = [0] * (N + 1)

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                elif rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1
                return True
            return False

        cost = 0
        count = 0
        for w, u, v in subset:
            if union(u, v):
                cost = (cost + w) % K
                count += 1

        # Check if we have a spanning tree
        if count == N - 1:
            min_cost = min(min_cost, cost)

    print(min_cost)

if __name__ == "__main__":
    main()