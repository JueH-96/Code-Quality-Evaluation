import sys
from collections import defaultdict, deque

def find_strongest_programmer(N, M, relations):
    # Create a graph to represent the superiority relations
    graph = defaultdict(list)
    indegree = [0] * (N + 1)

    for A, B in relations:
        graph[A].append(B)
        indegree[B] += 1

    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(1, N + 1) if indegree[i] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # Check if we have a unique strongest programmer
    if len(topo_order) == N and indegree[topo_order[-1]] == 0:
        return topo_order[-1]
    else:
        return -1

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    relations = []

    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        relations.append((A, B))
        index += 2

    result = find_strongest_programmer(N, M, relations)
    print(result)

if __name__ == "__main__":
    main()