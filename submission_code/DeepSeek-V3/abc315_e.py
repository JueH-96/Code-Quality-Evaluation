from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for i in range(1, N+1):
        C_i = int(data[index])
        index += 1
        for j in range(C_i):
            P = int(data[index])
            graph[P].append(i)
            in_degree[i] += 1
            index += 1
    
    # Perform topological sort
    queue = deque()
    for i in range(1, N+1):
        if in_degree[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Now, we need to find the minimal set of books to read to read book 1
    # We can perform a BFS from book 1, following the dependencies in reverse
    # to find all books that are required to read book 1
    
    required = set()
    queue = deque()
    queue.append(1)
    required.add(1)
    
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in required:
                required.add(v)
                queue.append(v)
    
    # Now, we need to find the order in which to read these books
    # We can use the topological order and filter out the required books
    # and then print them in the order they appear in the topological sort
    
    result = []
    for book in order:
        if book in required and book != 1:
            result.append(book)
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()