import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    edges = []
    for i in range(1, N):
        u, v = int(data[2 * i - 1]), int(data[2 * i])
        edges.append((u, v))

    # Build the tree
    tree = [[] for _ in range(N + 1)]
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)

    # Find all pairs of non-adjacent vertices
    candidates = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if j not in tree[i]:
                candidates.append((i, j))

    # Check each candidate pair
    count = 0
    for u, v in candidates:
        # Check if adding edge (u, v) creates a cycle with all vertices of degree 3
        visited = [False] * (N + 1)
        if dfs(tree, u, v, -1, visited):
            count += 1

    print(count)

def dfs(tree, u, v, parent, visited):
    visited[u] = True
    for neighbor in tree[u]:
        if neighbor == parent:
            continue
        if neighbor == v:
            return all(len(tree[node]) == 3 for node in visited if visited[node])
        if not visited[neighbor] and dfs(tree, neighbor, v, u, visited):
            return True
    return False

if __name__ == "__main__":
    solve()