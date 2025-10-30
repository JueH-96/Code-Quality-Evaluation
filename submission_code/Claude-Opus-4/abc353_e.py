# YOUR CODE HERE
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

def insert_and_count(root, s):
    node = root
    contribution = 0
    
    for char in s:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
        contribution += node.count
        node.count += 1
    
    return contribution

n = int(input())
strings = input().split()

root = TrieNode()
total = 0

for s in strings:
    total += insert_and_count(root, s)

print(total)