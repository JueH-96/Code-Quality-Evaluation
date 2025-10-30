import sys
from collections import deque

class TrieNode:
    __slots__ = ('children', 'count')
    def __init__(self):
        self.children = dict()
        self.count = 0

def main():
    n = int(sys.stdin.readline())
    strings = sys.stdin.readline().split()
    root = TrieNode()
    for s in strings:
        current = root
        for c in s:
            if c not in current.children:
                current.children[c] = TrieNode()
            current = current.children[c]
            current.count += 1
    total = 0
    queue = deque()
    for child in root.children.values():
        queue.append(child)
    while queue:
        node = queue.popleft()
        total += (node.count * (node.count - 1)) // 2
        for child in node.children.values():
            queue.append(child)
    print(total)

if __name__ == "__main__":
    main()