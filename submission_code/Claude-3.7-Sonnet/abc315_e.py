from collections import deque

def main():
    N = int(input())
    
    # Prerequisites for each book
    prereqs = [[] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        line = list(map(int, input().split()))
        C_i = line[0]
        prereqs[i] = line[1:C_i+1]
    
    # Find books required to read book 1
    required = set()
    
    def dfs(book):
        for p in prereqs[book]:
            if p not in required:
                required.add(p)
                dfs(p)
    
    dfs(1)
    
    # Create a graph of dependencies among required books
    graph = [[] for _ in range(N + 1)]
    in_degree = [0] * (N + 1)
    
    for book in required:
        for p in prereqs[book]:
            if p in required:
                graph[p].append(book)
                in_degree[book] += 1
    
    # Perform topological sort
    q = deque([book for book in required if in_degree[book] == 0])
    result = []
    
    while q:
        book = q.popleft()
        result.append(book)
        for next_book in graph[book]:
            in_degree[next_book] -= 1
            if in_degree[next_book] == 0:
                q.append(next_book)
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()