import sys
from collections import defaultdict, deque

def find_leaves(tree, n):
    leaves = []
    for i in range(1, n + 1):
        if len(tree[i]) == 1:
            leaves.append(i)
    return leaves

def remove_leaf(tree, leaf):
    for node in tree[leaf]:
        tree[node].remove(leaf)
    del tree[leaf]

def bfs_distance(tree, start, end):
    visited = set()
    queue = deque([(start, 0)])
    while queue:
        node, distance = queue.popleft()
        if node == end:
            return distance
        visited.add(node)
        for neighbor in tree[node]:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    return -1

def maximize_score(n, edges):
    tree = defaultdict(set)
    for a, b in edges:
        tree[a].add(b)
        tree[b].add(a)

    operations = []
    while len(tree) > 0:
        leaves = find_leaves(tree, n)
        max_distance = -1
        chosen_leaves = None
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                distance = bfs_distance(tree, leaves[i], leaves[j])
                if distance > max_distance:
                    max_distance = distance
                    chosen_leaves = (leaves[i], leaves[j])
        if chosen_leaves:
            operations.append(chosen_leaves)
            remove_leaf(tree, chosen_leaves[0])
            remove_leaf(tree, chosen_leaves[1])
    return operations

def main():
    n = int(sys.stdin.readline().strip())
    edges = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(n - 1)]
    operations = maximize_score(n, edges)
    for op in operations:
        print(op[0], op[1])

if __name__ == "__main__":
    main()