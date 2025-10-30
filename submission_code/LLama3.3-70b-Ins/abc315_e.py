from collections import defaultdict, deque

def solve():
    N = int(input())
    graph = defaultdict(list)
    indegree = [0] * (N + 1)

    for i in range(1, N + 1):
        C, *P = map(int, input().split())
        for j in P:
            graph[j].append(i)
            indegree[i] += 1

    queue = deque([i for i in range(1, N + 1) if indegree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        if node != 1:
            result.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    print(*result)

if __name__ == "__main__":
    solve()