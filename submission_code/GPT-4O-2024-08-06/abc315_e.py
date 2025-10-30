# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    
    from collections import defaultdict, deque
    
    # Graph adjacency list and in-degree count
    graph = defaultdict(list)
    in_degree = [0] * (N + 1)
    
    # Read the input and construct the graph
    for i in range(1, N + 1):
        line = list(map(int, data[i].split()))
        C_i = line[0]
        prerequisites = line[1:]
        for pre in prerequisites:
            graph[pre].append(i)
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    queue = deque()
    # We start with books that have no prerequisites
    for i in range(1, N + 1):
        if in_degree[i] == 0:
            queue.append(i)
    
    # Result list to store the order of books to read
    result = []
    
    while queue:
        book = queue.popleft()
        result.append(book)
        for neighbor in graph[book]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # We need to output the books required to read book 1, excluding book 1 itself
    # Find the position of book 1 in the result list
    book_1_index = result.index(1)
    # Output the books before book 1
    print(" ".join(map(str, result[:book_1_index])))