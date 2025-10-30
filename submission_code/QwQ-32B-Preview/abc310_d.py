def main():
    import sys

    # Read input
    input = sys.stdin.read().splitlines()
    N, T, M = map(int, input[0].split())
    incompatible_pairs = []
    for i in range(1, 1 + M):
        A, B = map(int, input[i].split())
        incompatible_pairs.append((A, B))

    # Initialize incompatible list as a list of sets
    incompatible = [set() for _ in range(N)]
    for A, B in incompatible_pairs:
        incompatible[A - 1].add(B - 1)
        incompatible[B - 1].add(A - 1)

    # Backtracking variables
    team_assignments = [-1] * N
    assigned_teams = set()
    count = 0

    def backtrack(player):
        nonlocal count
        if player == N:
            if len(assigned_teams) == T:
                count += 1
            return
        for team in range(T):
            valid = True
            for incompatible_player in incompatible[player]:
                if team_assignments[incompatible_player] == team:
                    valid = False
                    break
            if valid:
                team_assignments[player] = team
                if team not in assigned_teams:
                    assigned_teams.add(team)
                backtrack(player + 1)
                team_assignments[player] = -1
                if team in assigned_teams and team not in team_assignments:
                    assigned_teams.remove(team)

    # Start backtracking from the first player
    backtrack(0)
    print(count)

if __name__ == "__main__":
    main()