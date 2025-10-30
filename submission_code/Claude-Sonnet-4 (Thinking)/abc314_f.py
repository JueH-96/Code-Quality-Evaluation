from fractions import Fraction

n = int(input())
matches = []
for _ in range(n-1):
    p, q = map(int, input().split())
    matches.append((p-1, q-1))  # Convert to 0-indexed

# Initially, each player is on their own team
teams = {i: [i] for i in range(n)}
player_to_team = {i: i for i in range(n)}
expected_wins = [Fraction(0)] * n

for p, q in matches:
    team_p = player_to_team[p]
    team_q = player_to_team[q]
    
    size_p = len(teams[team_p])
    size_q = len(teams[team_q])
    
    prob_p = Fraction(size_p, size_p + size_q)
    prob_q = Fraction(size_q, size_p + size_q)
    
    # Add win probabilities to all players in the respective teams
    for player in teams[team_p]:
        expected_wins[player] += prob_p
    for player in teams[team_q]:
        expected_wins[player] += prob_q
    
    # Merge the teams
    for player in teams[team_q]:
        player_to_team[player] = team_p
    teams[team_p].extend(teams[team_q])
    del teams[team_q]

# Convert fractions to modular representation
MOD = 998244353
result = []
for frac in expected_wins:
    numerator = frac.numerator % MOD
    denominator = frac.denominator % MOD
    inv_denominator = pow(denominator, MOD-2, MOD)
    result.append((numerator * inv_denominator) % MOD)

print(' '.join(map(str, result)))