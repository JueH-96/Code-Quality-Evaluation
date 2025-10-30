from collections import deque

def solve():
    n = int(input())
    prerequisites = []
    for _ in range(n):
        line = list(map(int, input().split()))
        prerequisites.append(line[1:])

    required_books = set()
    queue = deque([1])
    visited = {1}

    while queue:
        current_book = queue.popleft()
        for i in range(1, n + 1):
            if current_book in prerequisites[i - 1]:
                if i not in visited:
                    visited.add(i)
                    queue.append(i)

    required_books = visited - {1}

    if not required_books:
        print()
        return

    adj = {book: [] for book in required_books}
    in_degree = {book: 0 for book in required_books}

    for book in required_books:
        for prereq in prerequisites[book - 1]:
            if prereq in required_books:
                adj[prereq].append(book)
                in_degree[book] += 1

    queue = deque([book for book in required_books if in_degree[book] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    print(*result)

solve()