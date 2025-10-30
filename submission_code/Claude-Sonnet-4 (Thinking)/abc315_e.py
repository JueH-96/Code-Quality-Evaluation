from collections import deque, defaultdict

n = int(input())

# Read prerequisites for each book
prerequisites = {}
for i in range(1, n + 1):
    line = list(map(int, input().split()))
    c = line[0]
    if c > 0:
        prereqs = line[1:]
    else:
        prereqs = []
    prerequisites[i] = prereqs

# Find all books required to read book 1
required = set()
queue = deque([1])

while queue:
    book = queue.popleft()
    for prereq in prerequisites[book]:
        if prereq not in required:
            required.add(prereq)
            queue.append(prereq)

# Build adjacency list for topological sorting
adj = defaultdict(list)
in_degree = defaultdict(int)

for book in required:
    in_degree[book] = 0

for book in required:
    for prereq in prerequisites[book]:
        if prereq in required:
            adj[prereq].append(book)
            in_degree[book] += 1

# Topological sort
queue = deque()
for book in required:
    if in_degree[book] == 0:
        queue.append(book)

result = []
while queue:
    book = queue.popleft()
    result.append(book)
    
    for neighbor in adj[book]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

print(*result)