def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    deps = [[] for _ in range(n + 1)]
    for i in range(1, n+1):
        c = int(next(it))
        for _ in range(c):
            prerequisite = int(next(it))
            deps[i].append(prerequisite)
            
    # We want the minimal set of books that must be read to read book 1.
    # That is exactly the dependency closure starting from book 1.
    # And then we output a topological ordering (order in which prerequisites come first)
    
    visited = [False] * (n+1)
    order = []
    
    def dfs(u):
        visited[u] = True
        for v in deps[u]:
            if not visited[v]:
                dfs(v)
        order.append(u)
    
    dfs(1)
    # order now is in postorder reverse it to get a valid ordering
    order.reverse()
    
    # Remove book 1 from the printed order
    result = [str(book) for book in order if book != 1]
    sys.stdout.write(" ".join(result))
    
if __name__ == '__main__':
    main()