def solve():
    n = int(input())
    dependencies = []
    for _ in range(n):
        line = list(map(int, input().split()))
        dependencies.append(line[1:])

    required_books = set()
    stack = [1]
    
    while stack:
        book = stack.pop()
        for pre_book in dependencies[book - 1]:
            if pre_book not in required_books:
                required_books.add(pre_book)
                stack.append(pre_book)

    
    reading_order = []
    visited = set()
    
    def topological_sort(book):
        visited.add(book)
        for pre_book in dependencies[book-1]:
            if pre_book in required_books and pre_book not in visited:
                topological_sort(pre_book)
        reading_order.append(book)

    for book in required_books:
        if book not in visited:
            topological_sort(book)
    
    reading_order.reverse()
    print(*reading_order)

solve()