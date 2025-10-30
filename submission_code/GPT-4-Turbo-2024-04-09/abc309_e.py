import sys
input = sys.stdin.read
from collections import defaultdict, deque

def solve():
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    M = int(data[idx + 1])
    idx += 2
    
    parents = [0] * (N + 1)
    for i in range(2, N + 1):
        parents[i] = int(data[idx])
        idx += 1
    
    insurances = []
    for _ in range(M):
        x = int(data[idx])
        y = int(data[idx + 1])
        idx += 2
        insurances.append((x, y))
    
    # Build the tree from parent information
    tree = defaultdict(list)
    for child in range(2, N + 1):
        parent = parents[child]
        tree[parent].append(child)
    
    # To keep track of coverage using a lazy propagation approach
    covered = [0] * (N + 1)
    
    # Process each insurance
    for x, y in insurances:
        covered[x] += 1
        if y + 1 < len(covered):  # If there's a valid "out of scope" for this coverage
            covered[min(N + 1, x + y + 1)] -= 1
    
    # Use BFS to propagate the coverage down the tree
    queue = deque([1])
    current_coverage = 0
    visited = [False] * (N + 1)
    
    while queue:
        node = queue.popleft()
        visited[node] = True
        current_coverage += covered[node]
        covered[node] = current_coverage
        
        for child in tree[node]:
            if not visited[child]:
                queue.append(child)
        
        # After processing this node, reduce the current coverage by the end mark
        current_coverage -= covered[min(N + 1, node + y + 1)]
    
    # Count the number of people covered by at least one insurance
    result = sum(1 for i in range(1, N + 1) if covered[i] > 0)
    
    print(result)