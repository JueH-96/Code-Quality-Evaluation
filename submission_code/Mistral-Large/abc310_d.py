import sys
from itertools import combinations

def can_form_teams(players, incompatible_pairs, T):
    for team_combination in combinations(range(1, N), T - 1):
        teams = [set(team_combination)]
        remaining_players = set(players) - set(team_combination)
        if not remaining_players:
            continue
        teams.append(remaining_players)

        valid = True
        for a, b in incompatible_pairs:
            team_a = team_b = None
            for team in teams:
                if a in team:
                    team_a = team
                if b in team:
                    team_b = team
            if team_a == team_b:
                valid = False
                break

        if valid:
            return True
    return False

def count_valid_divisions(N, T, M, incompatible_pairs):
    players = list(range(1, N + 1))
    count = 0

    for division in combinations(range(1, N), T - 1):
        teams = [set(division)]
        remaining_players = set(players) - set(division)
        if not remaining_players:
            continue
        teams.append(remaining_players)

        valid = True
        for a, b in incompatible_pairs:
            team_a = team_b = None
            for team in teams:
                if a in team:
                    team_a = team
                if b in team:
                    team_b = team
            if team_a == team_b:
                valid = False
                break

        if valid:
            count += 1

    return count

input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
M = int(data[2])

incompatible_pairs = []
index = 3
for _ in range(M):
    A = int(data[index])
    B = int(data[index + 1])
    incompatible_pairs.append((A, B))
    index += 2

result = count_valid_divisions(N, T, M, incompatible_pairs)
print(result)