import sys
from collections import deque

class TrieNode:
    __slots__ = ['children', 'count']
    def __init__(self):
        self.children = dict()
        self.count = 0

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]
    
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
        queue.append((child, 1))
    
    while queue:
        node, depth = queue.popleft()
        sum_child = 0
        for child in node.children.values():
            sum_child += child.count * (child.count - 1) // 2
        current_contribution = depth * ( (node.count * (node.count - 1) // 2) - sum_child )
        total += current_contribution
        for child in node.children.values():
            queue.append((child, depth + 1))
    
    print(total)

if __name__ == "__main__":
    main()