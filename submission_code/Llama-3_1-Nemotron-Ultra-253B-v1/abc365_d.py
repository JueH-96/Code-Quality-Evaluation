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

# Initialize DP for the first move
dp_prev = {}
for m in allowed_moves[0]:
    win = 1 if (m == 'R' and s[0] == 'S') or (m == 'S' and s[0] == 'P') or (m == 'P' and s[0] == 'R') else 0
    dp_prev[m] = win

for i in range(1, n):
    current_moves = allowed_moves[i]
    current_dp = {}
    for m in current_moves:
        max_prev = 0
        # Check previous allowed moves (from i-1)
        for prev_m in allowed_moves[i-1]:
            if prev_m != m:
                if dp_prev[prev_m] > max_prev:
                    max_prev = dp_prev[prev_m]
        # Calculate current win
        win = 1 if (m == 'R' and s[i] == 'S') or (m == 'S' and s[i] == 'P') or (m == 'P' and s[i] == 'R') else 0
        current_dp[m] = max_prev + win
    dp_prev = current_dp

print(max(dp_prev.values()))