class TrieNode:
    __slots__ = ['children', 'count']
    def __init__(self):
        self.children = {}
        self.count = 0

import sys
def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print(0)
        return
    n = int(data[0].strip())
    strings = data[1].split()
    if len(strings) > n:
        strings = strings[:n]
    
    root = TrieNode()
    for s in strings:
        node = root
        for c in s:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.count += 1
            
    stack = list(root.children.values())
    ans = 0
    while stack:
        node = stack.pop()
        if node.count >= 2:
            ans += node.count * (node.count - 1) // 2
        stack.extend(node.children.values())
        
    print(ans)

if __name__ == "__main__":
    main()