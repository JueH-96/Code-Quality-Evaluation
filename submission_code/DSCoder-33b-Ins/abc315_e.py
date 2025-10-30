from collections import deque

def solve():
    N = int(input())
    graph = [[] for _ in range(N+1)]
    indegree = [0]*(N+1)
    for i in range(1, N+1):
        books = list(map(int, input().split()))
        for j in range(1, len(books)):
            graph[books[j]].append(i)
            indegree[i] += 1

    queue = deque()
    for i in range(1, N+1):
        if indegree[i] == 0:
            queue.append(i)

    order = []
    while queue:
        book = queue.popleft()
        order.append(book)
        for next_book in graph[book]:
            indegree[next_book] -= 1
            if indegree[next_book] == 0:
                queue.append(next_book)

    print(*order[::-1])

solve()