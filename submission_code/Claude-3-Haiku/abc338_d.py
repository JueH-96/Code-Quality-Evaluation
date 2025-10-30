from collections import defaultdict

def min_tour_length(N, M, X):
    # Create a graph to represent the islands and bridges
    graph = defaultdict(list)
    for i in range(1, N+1):
        graph[i].append((i+1) % N)
        graph[i+1 % N].append(i)

    # Find the minimum tour length
    min_length = float('inf')
    for i in range(1, N):
        # Remove the i-th bridge
        del graph[i][0]
        del graph[i+1 % N][0]

        # Find the shortest tour that visits all the islands
        tour = []
        visited = set()
        queue = [(X[0], 0)]
        while queue:
            island, length = queue.pop(0)
            if island in visited:
                continue
            visited.add(island)
            tour.append(island)
            if len(tour) == len(X):
                min_length = min(min_length, length)
                break
            for neighbor in graph[island]:
                queue.append((neighbor, length + 1))

        # Restore the i-th bridge
        graph[i].append((i+1) % N)
        graph[i+1 % N].append(i)

    return min_length

# Read the input
N, M = map(int, input().split())
X = list(map(int, input().split()))

# Solve the problem
print(min_tour_length(N, M, X))