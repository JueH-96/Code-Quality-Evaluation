import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

parent = list(map(int, data[2:2 + N - 1]))
insurances = [tuple(map(int, data[2 + N - 1 + 2 * i:2 + N - 1 + 2 * i + 2])) for i in range(M)]

# Build the tree
tree = defaultdict(list)
for i in range(2, N + 1):
    tree[parent[i - 2]].append(i)

# Process each insurance
covered = set()
for x, y in insurances:
    queue = deque([(x, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth > y:
            break
        covered.add(node)
        for child in tree[node]:
            queue.append((child, depth + 1))

print(len(covered))