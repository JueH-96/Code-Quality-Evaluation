def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    strings = data[1:]
    
    # Define a TrieNode with a count and dictionary of children.
    class TrieNode:
        __slots__ = ('count', 'children')
        def __init__(self):
            self.count = 0  # number of strings passing through this node
            self.children = {}  # mapping from char to TrieNode
    
    # Create the root of the trie.
    root = TrieNode()
    
    # Insert each string into the trie.
    for s in strings:
        node = root
        for ch in s:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.count += 1  # update count for every node visited
    
    # Our answer is the sum over every non-root node of (node.count choose 2)
    # because every such node contributes 1 for each pair that passes through it.
    ans = 0
    # Use DFS (stack) to traverse all nodes of the trie starting from children of root.
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        if node.count > 1:
            ans += node.count * (node.count - 1) // 2
        for child in node.children.values():
            stack.append(child)
    
    # Print the answer.
    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()