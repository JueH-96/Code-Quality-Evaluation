# YOUR CODE HERE
from collections import deque, defaultdict

def find_required_books(n, prerequisites):
    # Build adjacency list of dependencies
    deps = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        deps[i] = prerequisites[i]
    
    # Find all books required to read book 1 using BFS/DFS
    required = set()
    visited = set()
    
    def dfs(book):
        if book in visited:
            return
        visited.add(book)
        if book != 1:
            required.add(book)
        for dep in deps[book]:
            dfs(dep)
    
    dfs(1)
    
    # Now perform topological sort on the required books
    # Calculate in-degree for required books only
    in_degree = defaultdict(int)
    adj = defaultdict(list)
    
    for book in required:
        for dep in deps[book]:
            if dep in required:
                in_degree[book] += 1
                adj[dep].append(book)
    
    # Also need to count dependencies from book 1
    for dep in deps[1]:
        if dep in required:
            in_degree[1] += 1
            adj[dep].append(1)
    
    # Find books with no dependencies among required books
    queue = deque()
    for book in required:
        if in_degree[book] == 0:
            queue.append(book)
    
    result = []
    while queue:
        book = queue.popleft()
        result.append(book)
        
        for next_book in adj[book]:
            in_degree[next_book] -= 1
            if in_degree[next_book] == 0 and next_book in required:
                queue.append(next_book)
    
    return result

# Read input
n = int(input())
prerequisites = [[] for _ in range(n + 1)]

for i in range(1, n + 1):
    line = list(map(int, input().split()))
    c = line[0]
    if c > 0:
        prerequisites[i] = line[1:c+1]

# Find and print the result
result = find_required_books(n, prerequisites)
print(' '.join(map(str, result)))