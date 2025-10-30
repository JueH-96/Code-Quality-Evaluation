def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N - 1):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        edges.append((A, B))
        index += 2
    C = list(map(int, data[index:index + N]))
    
    from collections import defaultdict
    
    # Build the tree
    tree = defaultdict(list)
    for A, B in edges:
        tree[A].append(B)
        tree[B].append(A)
    
    # Initial DFS to calculate f(1) and subtree sizes
    f = [0] * N
    subtree_size = [0] * N
    visited = [False] * N
    
    def dfs1(node, parent):
        visited[node] = True
        subtree_size[node] = 1
        total_cost = 0
        for neighbor in tree[node]:
            if neighbor == parent:
                continue
            if not visited[neighbor]:
                cost = dfs1(neighbor, node)
                total_cost += cost + C[neighbor] * subtree_size[neighbor]
                subtree_size[node] += subtree_size[neighbor]
        return total_cost
    
    # Calculate f(1)
    f[0] = dfs1(0, -1)
    
    # Second DFS to calculate f(v) for all v using f(1)
    visited = [False] * N
    
    def dfs2(node, parent):
        visited[node] = True
        for neighbor in tree[node]:
            if neighbor == parent:
                continue
            if not visited[neighbor]:
                # Transition from f[node] to f[neighbor]
                f[neighbor] = f[node] + (sum(C) - 2 * subtree_size[neighbor] * C[neighbor])
                dfs2(neighbor, node)
    
    dfs2(0, -1)
    
    # Find the minimum f(v)
    result = min(f)
    print(result)