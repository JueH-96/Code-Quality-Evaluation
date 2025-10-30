from collections import defaultdict, deque

def solve():
    n = int(input())
    
    # Build dependency graph
    dependencies = {}
    reverse_deps = defaultdict(list)
    
    for i in range(1, n + 1):
        line = list(map(int, input().split()))
        c = line[0]
        deps = line[1:] if c > 0 else []
        dependencies[i] = deps
        
        # Build reverse dependency graph
        for dep in deps:
            reverse_deps[dep].append(i)
    
    # Find all books required to read book 1
    required = set()
    queue = deque([1])
    visited = set([1])
    
    while queue:
        book = queue.popleft()
        for dep in dependencies[book]:
            if dep not in visited:
                visited.add(dep)
                required.add(dep)
                queue.append(dep)
    
    # Topological sort on required books
    # Calculate in-degrees for required books only
    in_degree = defaultdict(int)
    graph = defaultdict(list)
    
    for book in required:
        for dep in dependencies[book]:
            if dep in required:
                graph[dep].append(book)
                in_degree[book] += 1
    
    # Initialize in_degree for all required books
    for book in required:
        if book not in in_degree:
            in_degree[book] = 0
    
    # Topological sort
    queue = deque()
    for book in required:
        if in_degree[book] == 0:
            queue.append(book)
    
    result = []
    while queue:
        book = queue.popleft()
        result.append(book)
        
        for neighbor in graph[book]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    print(' '.join(map(str, result)))

solve()