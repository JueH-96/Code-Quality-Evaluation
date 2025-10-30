from fractions import Fraction

N = int(input())
matches = []
for _ in range(N - 1):
    p, q = map(int, input().split())
    matches.append((p - 1, q - 1))  # 0-indexed

# Track which team each player is in
team_of = list(range(N))  # team_of[i] is the team player i is in
teams = [[i] for i in range(N)]  # teams[i] is the list of players in team i

# Expected wins for each player
expected_wins = [Fraction(0) for _ in range(N)]

MOD = 998244353

# Process each match
for p, q in matches:
    # Find the teams containing players p and q
    team_p = team_of[p]
    team_q = team_of[q]
    
    # Get the sizes of the teams
    size_p = len(teams[team_p])
    size_q = len(teams[team_q])
    
    # Calculate probabilities
    prob_p = Fraction(size_p, size_p + size_q)
    prob_q = Fraction(size_q, size_p + size_q)
    
    # Update expected wins
    for player in teams[team_p]:
        expected_wins[player] += prob_p
    for player in teams[team_q]:
        expected_wins[player] += prob_q
    
    # Merge teams: merge team_q into team_p
    for player in teams[team_q]:
        team_of[player] = team_p
    teams[team_p].extend(teams[team_q])
    teams[team_q] = []

# Convert to modular arithmetic
result = []
for frac in expected_wins:
    num = frac.numerator % MOD
    den = frac.denominator % MOD
    # Find modular inverse of denominator using Fermat's little theorem
    inv_den = pow(den, MOD - 2, MOD)
    result.append((num * inv_den) % MOD)

print(' '.join(map(str, result)))