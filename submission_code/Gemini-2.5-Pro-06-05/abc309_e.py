import sys
from collections import deque

# This script is designed to be run directly, reading from stdin and writing to stdout.

# Read N (number of people) and M (number of insurance policies)
try:
    line = sys.stdin.readline()
    if line:
        N, M = map(int, line.split())
    else:
        # Handle empty input
        exit()
except (ValueError, IndexError):
    # Handle malformed input
    exit()

# Step 1: Build the tree structure using an adjacency list.
# adj[i] will store the children of person i. People are 1-indexed.
adj = [[] for _ in range(N + 1)]
if N > 1:
    parents = list(map(int, sys.stdin.readline().split()))
    # The i-th element of `parents` (0-indexed) is the parent of person i+2.
    for i, p in enumerate(parents):
        child_node = i + 2
        parent_node = p
        adj[parent_node].append(child_node)

# Step 2: Compute the depth of each person using Breadth-First Search (BFS).
# The depth is the distance from the root (person 1).
depth = [-1] * (N + 1)
# A deque is used for an efficient queue implementation.
queue = deque([(1, 0)])  # (node, current_depth)
depth[1] = 0
while queue:
    u, d = queue.popleft()
    for v in adj[u]:
        depth[v] = d + 1
        queue.append((v, d + 1))

# Step 3: Process insurance policies.
# For each person, we find the maximum "coverage limit" from policies they bought.
# A policy (x, y) has a coverage limit of `depth[x] + y`.
max_coverage_limit_at_node = [-1] * (N + 1)
for _ in range(M):
    x, y = map(int, sys.stdin.readline().split())
    coverage_limit = depth[x] + y
    max_coverage_limit_at_node[x] = max(max_coverage_limit_at_node[x], coverage_limit)
    
# Step 4: Propagate coverage limits and count covered people.
# We use an iterative Depth-First Search (DFS) to traverse the tree.
# This avoids recursion depth issues on large, chain-like trees.
covered_count = 0
# The stack stores tuples of (node, inherited_coverage_limit_from_ancestors).
stack = [(1, -1)]  # Start with root, which inherits no coverage (-1).

while stack:
    u, inherited_limit = stack.pop()
    
    # The effective limit for node `u` is the max of what's inherited
    # and what originates from `u` itself.
    my_limit = max_coverage_limit_at_node[u]
    effective_limit = max(inherited_limit, my_limit)

    # A person `u` is covered if their depth is within the effective limit.
    if depth[u] <= effective_limit:
        covered_count += 1

    # Pass the effective limit down to children.
    for v in adj[u]:
        stack.append((v, effective_limit))

# Step 5: Print the final answer.
print(covered_count)