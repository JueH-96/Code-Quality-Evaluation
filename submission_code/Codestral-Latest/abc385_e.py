import sys
from collections import defaultdict, deque

def read_input():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    edges = []
    for i in range(1, 2*N-1, 2):
        u = int(data[i])
        v = int(data[i+1])
        edges.append((u, v))
    return N, edges

def build_tree(edges):
    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)
    return tree

def bfs(tree, start):
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

def find_snowflake_tree(tree, N):
    # Find the centroid of the tree
    centroid = None
    for node in tree:
        if len(bfs(tree, node)) == N:
            centroid = node
            break

    # Find the children of the centroid
    children = tree[centroid]

    # Check if the tree can be transformed into a snowflake tree
    if len(children) < 2:
        return 0

    # Find the minimum number of vertices to delete
    min_deletions = N
    for child in children:
        subtree_size = len(bfs(tree, child))
        deletions = N - subtree_size - 1
        min_deletions = min(min_deletions, deletions)

    return min_deletions

def main():
    N, edges = read_input()
    tree = build_tree(edges)
    result = find_snowflake_tree(tree, N)
    print(result)

if __name__ == "__main__":
    main()