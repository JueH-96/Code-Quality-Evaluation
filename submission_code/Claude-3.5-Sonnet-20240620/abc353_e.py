# YOUR CODE HERE
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

def calculate_sum(root):
    return sum(node.count * (node.count - 1) // 2 for node in traverse(root))

def traverse(node):
    yield node
    for child in node.children.values():
        yield from traverse(child)

N = int(input())
strings = input().split()

root = TrieNode()
for s in strings:
    insert(root, s)

print(calculate_sum(root))