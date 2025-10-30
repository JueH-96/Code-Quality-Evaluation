def modinv(x, mod):
  return pow(x, mod - 2, mod)

def solve():
  n = int(input())
  matches = []
  for _ in range(n - 1):
    matches.append(list(map(int, input().split())))

  mod = 998244353
  teams = [{i} for i in range(1, n + 1)]
  wins = [0] * n

  for p, q in matches:
    p_idx = -1
    q_idx = -1
    for i in range(len(teams)):
      if p in teams[i]:
        p_idx = i
      if q in teams[i]:
        q_idx = i

    team_p = teams[p_idx]
    team_q = teams[q_idx]
    
    prob_p_wins = (len(team_p) * modinv(len(team_p) + len(team_q), mod)) % mod
    prob_q_wins = (len(team_q) * modinv(len(team_p) + len(team_q), mod)) % mod

    for player in team_p:
      wins[player - 1] = (wins[player - 1] + prob_p_wins) % mod
    for player in team_q:
      wins[player - 1] = (wins[player - 1] + prob_q_wins) % mod

    teams[p_idx] = teams[p_idx].union(teams[q_idx])
    teams.pop(q_idx)

  print(*wins)

solve()