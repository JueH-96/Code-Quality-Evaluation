# YOUR CODE HERE
def solve():
    n = int(input())
    dependencies = []
    for _ in range(n):
        line = list(map(int, input().split()))
        dependencies.append(line[1:])

    required_books = set()
    q = dependencies[0]
    
    while q:
        next_q = []
        for book in q:
            if book not in required_books:
                required_books.add(book)
                for dep in dependencies[book-1]:
                    if dep not in required_books:
                        next_q.append(dep)
        q = next_q
        
    result = []
    
    
    def find_order(books):
        
        order = []
        visited = set()
        
        def dfs(book):
            if book in visited:
                return
            visited.add(book)
            for dep in dependencies[book-1]:
                if dep in books and dep not in visited:
                    dfs(dep)
            order.append(book)

        for book in books:
            if book not in visited:
                dfs(book)
        return order[::-1]

    result = find_order(required_books)
    print(*result)

solve()