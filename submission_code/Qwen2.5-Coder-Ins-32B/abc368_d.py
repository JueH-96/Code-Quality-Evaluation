# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    K = int(input[index + 1])
    index += 2
    edges = []
    for _ in range(N - 1):
        A = int(input[index])
        B = int(input[index + 1])
        edges.append((A, B))
        index += 2
    required_vertices = list(map(int, input[index:index + K]))
    
    # Build the tree
    tree = defaultdict(list)
    for A, B in edges:
        tree[A].append(B)
        tree[B].append(A)
    
    # Find the LCA (Lowest Common Ancestor) of all required vertices
    def find_lca(tree, root, required_vertices):
        parents = {root: None}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in tree[node]:
                if neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        
        def path_to_root(node):
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return path
        
        paths = [path_to_root(v) for v in required_vertices]
        min_len = min(len(p) for p in paths)
        for i in range(min_len):
            current = paths[0][i]
            if all(p[i] == current for p in paths):
                continue
            return paths[0][i - 1]
        return paths[0][min_len - 1]
    
    lca = find_lca(tree, 1, required_vertices)
    
    # Count the number of nodes in the subtree rooted at LCA that includes all required vertices
    def count_nodes(tree, root, required_set):
        if not tree[root]:
            return root in required_set, 1 if root in required_set else 0
        
        count = 0
        includes_required = root in required_set
        for neighbor in tree[root]:
            child_includes_required, child_count = count_nodes(tree, neighbor, required_set)
            includes_required |= child_includes_required
            count += child_count
        
        return includes_required, count if includes_required else 0
    
    required_set = set(required_vertices)
    _, result = count_nodes(tree, lca, required_set)
    print(result)

if __name__ == "__main__":
    main()