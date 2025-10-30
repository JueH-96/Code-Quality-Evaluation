# YOUR CODE HERE
import sys
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
            node.count += 1
    
    def get_common_prefix_sum(self):
        total = 0
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 0:
                total += node.count * (node.count - 1) // 2
            for char, child in node.children.items():
                stack.append((child, depth + 1))
        return total

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1:N+1]
    trie = Trie()
    for s in S:
        trie.insert(s)
    print(trie.get_common_prefix_sum())

if __name__ == "__main__":
    main()