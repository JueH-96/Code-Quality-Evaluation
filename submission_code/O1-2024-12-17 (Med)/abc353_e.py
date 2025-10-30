def main():
    import sys
    sys.setrecursionlimit(10**7)

    # Read inputs
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    strings = data[1:]

    # Trie node definition
    class TrieNode:
        __slots__ = ('children', 'count')
        def __init__(self):
            self.children = {}
            self.count = 0

    root = TrieNode()

    # Insert a string into the trie
    def insert(s):
        current = root
        for ch in s:
            if ch not in current.children:
                current.children[ch] = TrieNode()
            current = current.children[ch]
            current.count += 1

    # Build the trie with all strings
    for s in strings:
        insert(s)

    # We sum c*(c-1)//2 for each node, where c is the number of strings
    # that pass through that node.  This effectively adds +1 for each
    # level of common prefix for every pair of strings.
    from collections import deque
    queue = deque(root.children.values())
    answer = 0
    while queue:
        node = queue.pop()
        c = node.count
        answer += c * (c - 1) // 2  # c choose 2
        for child in node.children.values():
            queue.append(child)

    print(answer)

# Important: call main() to ensure the solution runs
if __name__ == "__main__":
    main()