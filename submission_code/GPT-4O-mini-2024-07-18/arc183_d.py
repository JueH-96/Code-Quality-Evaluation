def main():
    import sys
    from collections import defaultdict, deque

    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    edges = [tuple(map(int, line.split())) for line in data[1:N]]
    
    # Build the tree
    tree = defaultdict(list)
    for a, b in edges:
        tree[a].append(b)
        tree[b].append(a)

    # Find all leaves
    leaves = [i for i in range(1, N + 1) if len(tree[i]) == 1]

    # To maximize the score, we will pair the leaves from the ends of the tree
    pairs = []
    
    while len(leaves) > 1:
        # Take the first two leaves
        u = leaves.pop()
        v = leaves.pop()
        pairs.append((u, v))
        
        # Remove them from the tree
        for leaf in (u, v):
            for neighbor in tree[leaf]:
                tree[neighbor].remove(leaf)
                if len(tree[neighbor]) == 1:  # If neighbor becomes a leaf
                    leaves.append(neighbor)
            del tree[leaf]  # Remove the leaf from the tree

    # Output the pairs
    for x, y in pairs:
        print(x, y)

if __name__ == "__main__":
    main()