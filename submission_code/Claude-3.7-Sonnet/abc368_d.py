def main():
    # Read input
    N, K = map(int, input().split())
    
    # Create adjacency list representation of the tree
    tree = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, input().split())
        tree[a].append(b)
        tree[b].append(a)
    
    # Read marked vertices
    marked_vertices = list(map(int, input().split()))
    
    # Mark the specified vertices
    is_marked = [False] * (N+1)
    for v in marked_vertices:
        is_marked[v] = True
    
    # DFS to compute the minimum subtree
    included_in_subtree = [False] * (N+1)
    dfs(1, -1, tree, is_marked, included_in_subtree)
    
    # Count the number of vertices in the subtree (1-indexed)
    print(sum(included_in_subtree[1:]))

def dfs(node, parent, tree, is_marked, included_in_subtree):
    # Check if the current node is marked
    contains_marked = is_marked[node]
    
    # Check if any of the children contains a marked vertex
    for child in tree[node]:
        if child != parent:
            if dfs(child, node, tree, is_marked, included_in_subtree):
                contains_marked = True
    
    # If the current node or any of its children contains a marked vertex, include it in the subtree
    if contains_marked:
        included_in_subtree[node] = True
    
    return contains_marked

if __name__ == "__main__":
    main()