import itertools

def count_ways(N, T, M, incompatible_pairs):
    def is_valid(team_assignment):
        for a, b in incompatible_pairs:
            if team_assignment[a-1] == team_assignment[b-1]:
                return False
        return True

    count = 0
    for team_assignment in itertools.product(range(T), repeat=N):
        if is_valid(team_assignment):
            count += 1
    return count

# Read input
N, T, M = map(int, input().split())
incompatible_pairs = [tuple(map(int, input().split())) for _ in range(M)]

# Calculate and print the result
print(count_ways(N, T, M, incompatible_pairs))