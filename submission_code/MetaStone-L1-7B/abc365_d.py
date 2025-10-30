n = int(input())
s = input().strip()

win_moves = []
for c in s:
    if c == 'R':
        win = 'P'
    elif c == 'P':
        win = 'S'
    else:
        win = 'R'
    draw = c
    win_moves.append([win, draw])

dp_prev = {}
# Initialize for i=0
current_moves = win_moves[0]
winning_move = current_moves[0]
draw_move = current_moves[1]
dp_prev[winning_move] = 1
dp_prev[draw_move] = 0

for i in range(1, n):
    current_moves = win_moves[i]
    winning_move = current_moves[0]
    dp_current = {}
    for current_move in current_moves:
        max_prev = 0
        for prev_move in dp_prev:
            if prev_move != current_move:
                if dp_prev[prev_move] > max_prev:
                    max_prev = dp_prev[prev_move]
        add_1 = 1 if current_move == winning_move else 0
        dp_current[current_move] = max_prev + add_1
    dp_prev = dp_current.copy()

if dp_prev:
    print(max(dp_prev.values()))
else:
    print(0)