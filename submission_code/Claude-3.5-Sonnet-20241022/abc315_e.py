from collections import defaultdict, deque

def solve():
    N = int(input())
    
    # Read prerequisites for each book
    prereqs = []
    for _ in range(N):
        line = list(map(int, input().split()))
        if line[0] == 0:
            prereqs.append([])
        else:
            prereqs.append(line[1:])
    
    # Build graph of required books starting from book 1
    required = set()
    def dfs(book):
        if book in required:
            return
        required.add(book)
        for prereq in prereqs[book-1]:
            dfs(prereq)
    
    dfs(1)
    required.remove(1)  # Remove book 1 from required set
    
    # Build dependency graph
    graph = defaultdict(list)
    indegree = defaultdict(int)
    for book in required:
        for prereq in prereqs[book-1]:
            if prereq in required:
                graph[prereq].append(book)
                indegree[book] += 1
    
    # Topological sort
    result = []
    q = deque([book for book in required if indegree[book] == 0])
    
    while q:
        curr = q.popleft()
        result.append(curr)
        
        for next_book in graph[curr]:
            indegree[next_book] -= 1
            if indegree[next_book] == 0:
                q.append(next_book)
    
    print(*result)

solve()