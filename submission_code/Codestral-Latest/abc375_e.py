import sys
from itertools import product

input = sys.stdin.read
data = input().split()

n = int(data[0])
teams = [0, 0, 0]
strengths = [0] * n
team_assignments = [0] * n

for i in range(n):
    team_assignments[i] = int(data[2 * i + 1]) - 1
    strengths[i] = int(data[2 * i + 2])
    teams[team_assignments[i]] += strengths[i]

total_strength = sum(teams)
if total_strength % 3 != 0:
    print(-1)
else:
    target_strength = total_strength // 3
    min_switches = float('inf')

    for switches in product([0, 1, 2], repeat=n):
        new_teams = teams[:]
        for i in range(n):
            if switches[i] != 0:
                new_teams[team_assignments[i]] -= strengths[i]
                new_teams[(team_assignments[i] + switches[i]) % 3] += strengths[i]

        if new_teams[0] == new_teams[1] == new_teams[2] == target_strength:
            min_switches = min(min_switches, sum(switches))

    if min_switches == float('inf'):
        print(-1)
    else:
        print(min_switches)