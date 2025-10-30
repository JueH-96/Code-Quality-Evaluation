def modinv(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m

MOD = 998244353

N = int(input())
matches = []
for _ in range(N-1):
    p, q = map(int, input().split())
    matches.append((p-1, q-1))

# Initialize teams
teams = [{i} for i in range(N)]
team_of = list(range(N))

# Expected wins for each player
exp_wins = [0] * N

for match_idx, (p, q) in enumerate(matches):
    # Get teams of players p and q
    team1 = teams[team_of[p]]
    team2 = teams[team_of[q]]
    
    # Calculate probabilities
    a = len(team1)
    b = len(team2)
    p1 = a / (a + b)
    p2 = b / (a + b)
    
    # Update expected wins
    for player in team1:
        exp_wins[player] += p1
    for player in team2:
        exp_wins[player] += p2
        
    # Merge teams
    new_team = team1 | team2
    new_team_idx = team_of[p]
    teams[new_team_idx] = new_team
    for player in team2:
        team_of[player] = new_team_idx

# Convert to modulo
result = []
for x in exp_wins:
    num, den = x.as_integer_ratio()
    result.append((num * modinv(den, MOD)) % MOD)

print(*result)