beats = {'R': 'S', 'S': 'P', 'P': 'R'}

n = int(input())
s = input().strip()

allowed_moves = []
for c in s:
    if c == 'R':
        allowed_moves.append(['R', 'P'])
    elif c == 'P':
        allowed_moves.append(['P', 'S'])
    else:
        allowed_moves.append(['S', 'R'])

prev_dp = {}
for move in allowed_moves[0]:
    prev_dp[move] = 1 if beats[move] == s[0] else 0

for i in range(1, n):
    current_allowed = allowed_moves[i]
    new_dp = {}
    for c in current_allowed:
        max_prev = 0
        for p in prev_dp:
            if p != c and prev_dp[p] > max_prev:
                max_prev = prev_dp[p]
        new_dp[c] = max_prev + (1 if beats[c] == s[i] else 0)
    prev_dp = new_dp

print(max(prev_dp.values()))