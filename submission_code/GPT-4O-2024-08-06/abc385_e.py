def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N - 1):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        edges.append((u, v))
        index += 2
    
    from collections import defaultdict
    
    # Build the adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # To store the size of subtree rooted at each node
    subtree_size = [0] * N
    
    def dfs(node, parent):
        subtree_size[node] = 1
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            dfs(neighbor, node)
            subtree_size[node] += subtree_size[neighbor]
    
    # Run DFS from node 0 (or any node, since it's a tree)
    dfs(0, -1)
    
    # Function to calculate minimum deletions for a given root
    def min_deletions_for_root(root):
        # We need to find the best x and y such that 1 + x + x * y <= subtree_size[root]
        best_deletions = float('inf')
        for x in range(1, N):
            y = (subtree_size[root] - 1 - x) // x if x > 0 else 0
            if y < 0:
                continue
            if 1 + x + x * y <= subtree_size[root]:
                # Calculate deletions needed
                deletions = subtree_size[root] - (1 + x + x * y)
                best_deletions = min(best_deletions, deletions)
        return best_deletions
    
    # Calculate the minimum deletions for each node being the root
    min_deletions = float('inf')
    for node in range(N):
        min_deletions = min(min_deletions, min_deletions_for_root(node))
    
    print(min_deletions)