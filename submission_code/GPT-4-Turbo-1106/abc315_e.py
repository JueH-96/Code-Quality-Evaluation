from collections import deque

def topological_sort(graph, in_degree):
    order = []
    q = deque([node for node in range(len(in_degree)) if in_degree[node] == 0])

    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                q.append(neighbor)

    return order

def main():
    N = int(input())
    graph = [[] for _ in range(N)]
    in_degree = [0] * N

    for i in range(N):
        dependencies = list(map(int, input().split()))
        C_i = dependencies[0]
        for j in range(C_i):
            P_ij = dependencies[j + 1] - 1
            graph[P_ij].append(i)
            in_degree[i] += 1

    order = topological_sort(graph, in_degree)
    # Exclude book 1 (index 0) and convert to 1-based indexing
    result = [str(book + 1) for book in order if book != 0]
    print(' '.join(result))

if __name__ == "__main__":
    main()