from itertools import combinations
from collections import defaultdict

def count_divisions(N, T, M, incompatible_pairs):
    # Create a graph to represent the incompatible pairs
    graph = defaultdict(list)
    for a, b in incompatible_pairs:
        graph[a].append(b)
        graph[b].append(a)

    # Function to generate all possible divisions
    def generate_divisions(players, team_size):
        if not players:
            yield []
            return

        for i in range(1, min(team_size, len(players)) + 1):
            team = players[:i]
            for division in generate_divisions(players[i:], team_size):
                yield [team] + division

    # Count the number of valid divisions
    count = 0
    for division in generate_divisions(list(range(1, N + 1)), T):
        is_valid = True
        for a, b in incompatible_pairs:
            if a in division[b - 1] and b in division[a - 1]:
                is_valid = False
                break
        if is_valid:
            count += 1

    return count

# Read input from stdin
N, T, M = map(int, input().split())
incompatible_pairs = []
for _ in range(M):
    a, b = map(int, input().split())
    incompatible_pairs.append((a, b))

# Solve the problem and print the answer
print(count_divisions(N, T, M, incompatible_pairs))