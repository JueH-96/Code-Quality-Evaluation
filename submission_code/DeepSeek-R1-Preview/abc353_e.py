class Node:
    __slots__ = ('count', 'children', 'depth')
    def __init__(self, depth):
        self.count = 0
        self.children = {}
        self.depth = depth

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]

    root = Node(0)

    # Insert all strings into the trie
    for s in strings:
        node = root
        for c in s:
            if c not in node.children:
                node.children[c] = Node(node.depth + 1)
            node = node.children[c]
            node.count += 1

    total = 0
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            # Push children in reverse order to process them in order
            for child in reversed(list(node.children.values())):
                stack.append((child, False))
        else:
            if node.depth == 0:
                continue  # root contributes nothing
            c = node.count
            sum_child_pairs = 0
            for child in node.children.values():
                sum_child_pairs += child.count * (child.count - 1) // 2
            node_pairs = c * (c - 1) // 2
            contribution = (node_pairs - sum_child_pairs) * node.depth
            total += contribution

    print(total)

if __name__ == "__main__":
    main()