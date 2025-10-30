from collections import deque

def main():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        a, b = map(int, input().split())
        edges.append((a, b))
    
    # Build adjacency list for the tree
    tree = [[] for _ in range(n + 1)]
    for a, b in edges:
        tree[a].append(b)
        tree[b].append(a)
    
    operations = []
    removed = set()
    
    for _ in range(n // 2):
        # Find all current leaf nodes
        leaves = [i for i in range(1, n + 1) if i not in removed and sum(1 for neighbor in tree[i] if neighbor not in removed) == 1]
        
        max_distance = -1
        best_pair = None
        
        # Try all pairs of leaves and find the pair with the maximum distance
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                leaf1, leaf2 = leaves[i], leaves[j]
                distance = bfs_distance(tree, removed, leaf1, leaf2)
                
                if distance > max_distance:
                    max_distance = distance
                    best_pair = (leaf1, leaf2)
        
        operations.append(best_pair)
        removed.add(best_pair[0])
        removed.add(best_pair[1])
    
    # Print the operations
    for op in operations:
        print(f"{op[0]} {op[1]}")

def bfs_distance(tree, removed, start, end):
    """Calculate the shortest distance between start and end nodes in the tree,
    considering removed nodes."""
    queue = deque([start])
    visited = [False] * len(tree)
    visited[start] = True
    distance = [0] * len(tree)
    
    while queue:
        node = queue.popleft()
        if node == end:
            return distance[node]
        for neighbor in tree[node]:
            if not visited[neighbor] and neighbor not in removed:
                visited[neighbor] = True
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    
    return -1  # This should not happen in a connected tree

if __name__ == "__main__":
    main()