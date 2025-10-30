from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    idx = 1
    graph = defaultdict(list)
    for i in range(1, N+1):
        C_i = int(data[idx])
        idx += 1
        for j in range(C_i):
            P = int(data[idx])
            graph[i].append(P)
            idx += 1
    
    # Perform BFS to find all required books
    visited = set()
    queue = deque()
    queue.append(1)
    visited.add(1)
    required = []
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                required.append(neighbor)
    
    # Now, we need to find the order in which to read these books
    # We can perform a topological sort on the subgraph of required books
    # First, build the subgraph
    subgraph = defaultdict(list)
    in_degree = defaultdict(int)
    for book in required:
        for neighbor in graph[book]:
            if neighbor in visited:
                subgraph[book].append(neighbor)
                in_degree[neighbor] += 1
    
    # Initialize the queue with books that have no prerequisites
    queue = deque()
    for book in required:
        if in_degree[book] == 0:
            queue.append(book)
    
    order = []
    while queue:
        current = queue.popleft()
        if current != 1:
            order.append(current)
        for neighbor in subgraph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    print(' '.join(map(str, order)))

if __name__ == "__main__":
    main()