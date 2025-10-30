from collections import defaultdict

def find_strongest_programmer():
    # Read the number of programmers and the number of pieces of information
    N, M = map(int, input().split())

    # Create a graph to store the superiority relationships
    graph = defaultdict(list)
    for _ in range(M):
        A, B = map(int, input().split())
        graph[A].append(B)

    # Initialize a set to store the possible strongest programmers
    strongest_programmers = set(range(1, N + 1))

    # Iterate over the graph to find the possible strongest programmers
    for node in graph:
        for neighbor in graph[node]:
            if neighbor in strongest_programmers:
                strongest_programmers.remove(neighbor)

    # If there is only one possible strongest programmer, return it
    if len(strongest_programmers) == 1:
        return list(strongest_programmers)[0]
    else:
        return -1

# Print the result
print(find_strongest_programmer())