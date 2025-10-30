from collections import defaultdict

def find_confused_villagers(N, M, testimonies):
    # Create a graph where each node represents a villager
    # and edges represent testimonies
    graph = defaultdict(list)
    for a, b, c in testimonies:
        graph[a].append((b, c))

    # Initialize all villagers as not confused
    confused = [0] * N

    # Iterate through the testimonies and check for contradictions
    for a, b_c in graph.items():
        for b, c in b_c:
            for other_b, other_c in b_c:
                if b != other_b:
                    # If a testimony contradicts another, mark the villager as confused
                    if c != other_c:
                        confused[a - 1] = 1
                        confused[b - 1] = 1
                        confused[other_b - 1] = 1

    # Check if there is a valid set of confused villagers
    if any(confused):
        return ''.join(map(str, confused))
    else:
        return '-1'

# Read input
N, M = map(int, input().split())
testimonies = []
for _ in range(M):
    a, b, c = map(int, input().split())
    testimonies.append((a, b, c))

# Solve the problem
result = find_confused_villagers(N, M, testimonies)
print(result)