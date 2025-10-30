# YOUR CODE HERE
from itertools import combinations

def count_ways(N, T, M, pairs):
    def is_valid(teams):
        for a, b in pairs:
            for team in teams:
                if a in team and b in team:
                    return False
        return True

    players = list(range(1, N + 1))
    total_ways = 0

    for team_sizes in combinations(range(1, N), T - 1):
        team_sizes = list(team_sizes) + [N - sum(team_sizes)]
        if min(team_sizes) < 1:
            continue

        for teams in combinations(combinations(players, team_sizes[0]), 1):
            remaining_players = list(set(players) - set(teams[0]))
            if len(remaining_players) < sum(team_sizes[1:]):
                continue

            for next_teams in combinations(combinations(remaining_players, team_sizes[1]), 1):
                remaining_players = list(set(remaining_players) - set(next_teams[0]))
                if len(remaining_players) < sum(team_sizes[2:]):
                    continue

                for final_teams in combinations(combinations(remaining_players, team_sizes[2]), 1):
                    remaining_players = list(set(remaining_players) - set(final_teams[0]))
                    if len(remaining_players) < sum(team_sizes[3:]):
                        continue

                    current_teams = list(teams) + list(next_teams) + list(final_teams)
                    if len(remaining_players) > 0:
                        current_teams.append(tuple(remaining_players))

                    if is_valid(current_teams):
                        total_ways += 1

    return total_ways

import sys
input = sys.stdin.read().split()
N = int(input[0])
T = int(input[1])
M = int(input[2])
pairs = [(int(input[3 + 2 * i]), int(input[4 + 2 * i])) for i in range(M)]

print(count_ways(N, T, M, pairs))