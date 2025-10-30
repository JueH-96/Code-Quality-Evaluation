import sys

class TrieNode:
    __slots__ = ('children', 'count')
    def __init__(self):
        self.children = {}
        self.count = 0

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    strings = input[1:N+1]
    
    root = TrieNode()
    
    for s in strings:
        node = root
        for c in s:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.count += 1
    
    sum_ans = 0
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        sum_ans += node.count * (node.count - 1) // 2
        stack.extend(node.children.values())
    
    print(sum_ans)

if __name__ == "__main__":
    main()