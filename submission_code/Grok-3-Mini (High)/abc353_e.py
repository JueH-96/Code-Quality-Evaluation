import sys
from collections import deque

class TrieNode:
    def __init__(self):
        self.children = {}
        self.freq = 0

# Read input
data = sys.stdin.read().split()
N = int(data[0])
strings = data[1:N+1]

# Build Trie
root = TrieNode()
for s in strings:
    curr = root
    for char in s:
        if char not in curr.children:
            curr.children[char] = TrieNode()
        curr = curr.children[char]
        curr.freq += 1  # Increment frequency for each node along the path

# Compute sum using BFS
total_sum = 0
queue = deque()
queue.append(root)
while queue:
    node = queue.popleft()
    freq_node = node.freq
    total_sum += (freq_node * (freq_node - 1)) // 2
    for child in node.children.values():
        queue.append(child)

# Output the result
print(total_sum)