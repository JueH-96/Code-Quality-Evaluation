class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

n = int(input())
strings = input().split()

# Build trie
root = TrieNode()
for s in strings:
    node = root
    for char in s:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
        node.count += 1

# Calculate answer using DFS
def dfs(node):
    result = 0
    if node.count >= 2:
        result += node.count * (node.count - 1) // 2
    
    for child in node.children.values():
        result += dfs(child)
    
    return result

print(dfs(root))