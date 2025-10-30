from collections import defaultdict

def find_strongest_programmer(N, M, relations):
    # Create a graph to represent the superiority relations
    graph = defaultdict(list)
    for a, b in relations:
        graph[a].append(b)
        graph[b].append(a)
    
    # Find the in-degree of each node (programmer)
    in_degree = [0] * (N + 1)
    for a, b in relations:
        in_degree[b] += 1
    
    # Find the strongest programmer
    strongest = None
    for i in range(1, N + 1):
        if in_degree[i] == N - 1:
            if strongest is None:
                strongest = i
            else:
                # Multiple strongest programmers, return -1
                return -1
    
    return strongest if strongest else -1

# Read input
N, M = map(int, input().split())
relations = [list(map(int, input().split())) for _ in range(M)]

# Solve the problem
print(find_strongest_programmer(N, M, relations))