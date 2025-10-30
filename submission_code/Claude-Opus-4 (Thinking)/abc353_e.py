class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

def insert(root, s):
    node = root
    for char in s:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
        node.count += 1

def calculate_sum(node):
    total = 0
    if node.count >= 2:
        total += node.count * (node.count - 1) // 2
    for child in node.children.values():
        total += calculate_sum(child)
    return total

n = int(input())
strings = input().split()

root = TrieNode()
for s in strings:
    insert(root, s)

# Skip the root node (empty prefix)
total = 0
for child in root.children.values():
    total += calculate_sum(child)

print(total)