import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    dependents = [[] for _ in range(n + 1)]
    prerequisites = [[] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        parts = list(map(int, sys.stdin.readline().split()))
        c_i = parts[0]
        p_list = parts[1:]
        prerequisites[i] = p_list
        for p in p_list:
            dependents[p].append(i)
    
    # Collect required books
    is_required = [False] * (n + 1)
    queue = deque()
    
    # Initialize with prerequisites of 1
    for p in prerequisites[1]:
        if not is_required[p]:
            is_required[p] = True
            queue.append(p)
    
    # BFS to collect all dependencies
    while queue:
        x = queue.popleft()
        for p in prerequisites[x]:
            if not is_required[p]:
                is_required[p] = True
                queue.append(p)
    
    # Get list of required books (excluding 1)
    required_books = [i for i in range(1, n + 1) if is_required[i]]
    
    # Compute in_degree for each required book
    in_degree = {}
    for book in required_books:
        in_degree[book] = len(prerequisites[book])
    
    # Initialize queue for Kahn's algorithm
    q = deque()
    for book in required_books:
        if in_degree[book] == 0:
            q.append(book)
    
    topological_order = []
    while q:
        y = q.popleft()
        topological_order.append(y)
        for x in dependents[y]:
            if is_required[x]:
                in_degree[x] -= 1
                if in_degree[x] == 0:
                    q.append(x)
    
    print(' '.join(map(str, topological_order)))

if __name__ == '__main__':
    main()