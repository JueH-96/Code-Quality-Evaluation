n = int(input())
s = input().strip()
t = input().strip()

# First check if transformation is possible
char_map = {}
for i in range(n):
    if s[i] in char_map:
        if char_map[s[i]] != t[i]:
            print(-1)
            exit()
    else:
        char_map[s[i]] = t[i]

# Build graph of transformations needed
from collections import defaultdict, deque

# Find connected components using Union-Find
parent = list(range(26))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

# Group characters by their target
target_groups = defaultdict(list)
for char, target in char_map.items():
    target_groups[target].append(ord(char) - ord('a'))

# Union characters that map to the same target
for group in target_groups.values():
    for i in range(1, len(group)):
        union(group[0], group[i])

# Count operations for each component
used_chars = set(char_map.keys())
operations = 0

components = defaultdict(list)
for char in used_chars:
    char_idx = ord(char) - ord('a')
    root = find(char_idx)
    components[root].append(char)

for component in components.values():
    target = char_map[component[0]]
    if target in component:
        operations += len(component) - 1
    else:
        operations += len(component)

print(operations)