def solve():
    n, t, m = map(int, input().split())
    incompatible = []
    for _ in range(m):
        incompatible.append(tuple(map(int, input().split())))

    def is_valid_division(division):
        for a, b in incompatible:
            team_a = -1
            team_b = -1
            for i, team in enumerate(division):
                if a in team:
                    team_a = i
                if b in team:
                    team_b = i
            if team_a == team_b and team_a != -1:
                return False
        return True

    def count_valid_divisions(players, num_teams, current_division, start_player):
        if start_player == len(players):
            if len(current_division) == num_teams and all(len(team) > 0 for team in current_division):
                if is_valid_division(current_division):
                    return 1
                else:
                    return 0
            else:
                return 0
        
        count = 0
        for i in range(len(current_division)):
            new_division = [team[:] for team in current_division]
            new_division[i].append(players[start_player])
            count += count_valid_divisions(players, num_teams, new_division, start_player + 1)
        
        if len(current_division) < num_teams:
            new_division = [team[:] for team in current_division]
            new_division.append([players[start_player]])
            count += count_valid_divisions(players, num_teams, new_division, start_player + 1)
        
        return count

    players = list(range(1, n + 1))
    result = count_valid_divisions(players, t, [], 0)
    print(result)

solve()