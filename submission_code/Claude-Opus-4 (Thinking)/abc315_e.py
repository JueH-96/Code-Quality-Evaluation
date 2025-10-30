from collections import defaultdict, deque

# Read input
n = int(input())
prerequisites = defaultdict(list)

for i in range(1, n + 1):
    line = list(map(int, input().split()))
    c = line[0]
    if c > 0:
        prerequisites[i] = line[1:c+1]

# Find all books required to read book 1
required = set()
stack = [1]

while stack:
    book = stack.pop()
    for prereq in prerequisites[book]:
        if prereq not in required:
            required.add(prereq)
            stack.append(prereq)

# Build adjacency list and in-degree for the subgraph
graph = defaultdict(list)
in_degree = defaultdict(int)

for book in required:
    in_degree[book] = 0

for book in required:
    for prereq in prerequisites[book]:
        if prereq in required:
            graph[prereq].append(book)
            in_degree[book] += 1

# Kahn's algorithm for topological sort
queue = deque()
for book in required:
    if in_degree[book] == 0:
        queue.append(book)

result = []
while queue:
    book = queue.popleft()
    result.append(book)
    
    for next_book in graph[book]:
        in_degree[next_book] -= 1
        if in_degree[next_book] == 0:
            queue.append(next_book)

# Print result
print(' '.join(map(str, result)))