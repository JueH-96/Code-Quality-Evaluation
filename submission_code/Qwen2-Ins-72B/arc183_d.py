from collections import deque

def solve(N, edges):
    # Build the graph
    graph = [[] for _ in range(N)]
    for a, b in edges:
        graph[a-1].append(b-1)
        graph[b-1].append(a-1)

    # Initialize the queue with the first leaf node
    queue = deque([1])
    visited = [False] * N
    visited[0] = True

    # Find the farthest leaf node
    while queue:
        current = queue.popleft()
        farthest = current
        for neighbor in graph[current-1]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    queue = deque([farthest])
    visited = [False] * N
    visited[farthest] = True

    # Find the second farthest leaf node
    while queue:
        current = queue.popleft()
        second_farthest = current
        for neighbor in graph[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    # Initialize the answer list
    answer = []

    # Perform the operations
    while len(answer) < N//2:
        # Remove the farthest leaf node
        answer.append([farthest+1, second_farthest+1])
        graph[farthest].remove(second_farthest)
        graph[second_farthest].remove(farthest)

        # Find the new farthest leaf node
        farthest = second_farthest
        queue = deque([farthest])
        visited = [False] * N
        visited[farthest] = True
        while queue:
            current = queue.popleft()
            farthest = current
            for neighbor in graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        # Find the new second farthest leaf node
        queue = deque([farthest])
        visited = [False] * N
        visited[farthest] = True
        while queue:
            current = queue.popleft()
            second_farthest = current
            for neighbor in graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

    # Print the answer
    for a, b in answer:
        print(a, b)

N = int(input())
edges = [list(map(int, input().split())) for _ in range(N-1)]
solve(N, edges)