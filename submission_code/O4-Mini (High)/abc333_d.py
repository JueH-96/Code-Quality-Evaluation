import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # BFS from node 1 to establish parent relationships
    parent = [0] * (N+1)
    order = [1]
    parent[1] = 0
    i = 0
    while i < len(order):
        u = order[i]
        for w in adj[u]:
            if w != parent[u]:
                parent[w] = u
                order.append(w)
        i += 1

    # Compute subtree sizes in reverse BFS order (children before parents)
    subtree = [1] * (N+1)
    for v in reversed(order):
        p = parent[v]
        if p != 0:
            subtree[p] += subtree[v]

    # Among the neighbors of 1, find the largest subtree size
    best = 0
    for v in adj[1]:
        # Only consider those discovered as children of 1
        if parent[v] == 1 and subtree[v] > best:
            best = subtree[v]

    # To make node 1 a leaf, we remove all other subtrees:
    # total operations = (N - best)
    print(N - best)

if __name__ == "__main__":
    main()