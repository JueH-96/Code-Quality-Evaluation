import sys
import sys
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    from collections import defaultdict

    class TrieNode:
        __slots__ = ['children', 'count']
        def __init__(self):
            self.children = dict()
            self.count = 0

    root = TrieNode()

    input = sys.stdin.read().split()
    N = int(input[0])
    strings = input[1:]

    for s in strings:
        node = root
        node.count +=1
        for c in s:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.count +=1

    total = 0

    def dfs(node, depth):
        nonlocal total
        c = node.count
        pairs = c * (c -1) //2
        child_pairs =0
        for child in node.children.values():
            child_p = dfs(child, depth+1)
            child_pairs += child_p
        contrib = (pairs - child_pairs) * depth
        total += contrib
        return pairs

    dfs(root, 0)
    print(total)

if __name__ == "__main__":
    main()