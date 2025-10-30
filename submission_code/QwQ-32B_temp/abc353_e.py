import sys
from collections import deque

class TrieNode:
    __slots__ = ['children', 'count']
    def __init__(self):
        self.children = [None] * 26  # For each letter a-z
        self.count = 0

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    strings = input[1:N+1]

    root = TrieNode()
    for s in strings:
        current = root
        current.count += 1
        for c in s:
            idx = ord(c) - ord('a')
            if current.children[idx] is None:
                current.children[idx] = TrieNode()
            current = current.children[idx]
            current.count += 1

    total = 0
    queue = deque()
    # Initialize queue with root's children
    for i in range(26):
        child = root.children[i]
        if child is not None:
            queue.append(child)
    
    while queue:
        node = queue.popleft()
        total += node.count * (node.count - 1) // 2
        # Add all non-None children to the queue
        for i in range(26):
            child = node.children[i]
            if child is not None:
                queue.append(child)
    
    print(total)

if __name__ == "__main__":
    main()