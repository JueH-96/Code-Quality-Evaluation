import itertools

def solve():
    n, t, m = map(int, input().split())
    incompatible = set()
    for _ in range(m):
        u, v = map(int, input().split())
        incompatible.add(tuple(sorted((u, v))))

    count = 0
    for assignment_tuple in itertools.product(range(1, t + 1), repeat=n):
        assignment = list(assignment_tuple)

        # Check if each team has at least one player
        team_counts = [0] * t
        for team_id in assignment:
            team_counts[team_id - 1] += 1

        if any(count == 0 for count in team_counts):
            continue

        # Check for incompatible pairs
        is_valid = True
        for u, v in incompatible:
            if assignment[u - 1] == assignment[v - 1]:
                is_valid = False
                break
        if is_valid:
            count += 1
    print(count)

solve()