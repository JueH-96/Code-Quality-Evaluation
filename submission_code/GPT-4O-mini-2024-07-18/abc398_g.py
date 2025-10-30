def find_winner(N, M, edges):
    from collections import defaultdict, deque

    # Create adjacency list for the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # To find connected components and their sizes
    visited = [False] * (N + 1)
    component_sizes = []

    def bfs(start):
        queue = deque([start])
        visited[start] = True
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return size

    # Find all connected components
    for i in range(1, N + 1):
        if not visited[i]:
            component_size = bfs(i)
            component_sizes.append(component_size)

    # Calculate the total number of possible edges
    total_possible_edges = 0
    for size in component_sizes:
        total_possible_edges += size * (size - 1) // 2  # C(size, 2)

    # Current edges count
    current_edges_count = M

    # Remaining edges that can be added
    remaining_edges = total_possible_edges - current_edges_count

    # Determine the winner based on the parity of remaining edges
    if remaining_edges % 2 == 1:
        return "Aoki"
    else:
        return "Takahashi"

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, M = map(int, data[0].split())
    edges = [tuple(map(int, line.split())) for line in data[1:M + 1]]
    
    winner = find_winner(N, M, edges)
    print(winner)

if __name__ == "__main__":
    main()