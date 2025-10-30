from collections import defaultdict

class Node:
    def __init__(self):
        self.children = defaultdict(Node)
        self.count = 0

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
            node.count += 1

    def calculate_score(self):
        score = 0
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            score += node.count * depth
            for child in node.children.values():
                stack.append((child, depth + 1))
        return score

def calculate_longest_common_prefix_score(strings):
    trie = Trie()
    for string in strings:
        trie.insert(string)
    return trie.calculate_score()

# Read input
N = int(input())
strings = []
for _ in range(N):
    strings.append(input())

# Calculate and print the answer
print(calculate_longest_common_prefix_score(strings))