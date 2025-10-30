N = int(input())
S = input().strip()

win_against = {'P': 'S', 'R': 'P', 'S': 'R'}

def get_score(my_move, opponent_move):
    if my_move == win_against[opponent_move]:
        return 1
    elif my_move == opponent_move:
        return 0
    else:
        return -float('inf')

dp = {}

for move in ['R', 'P', 'S']:
    dp[(N-1, move)] = get_score(move, S[N-1])

for i in range(N-2, -1, -1):
    for move in ['R', 'P', 'S']:
        current_score = get_score(move, S[i])
        if current_score == -float('inf'):
            dp[(i, move)] = -float('inf')
        else:
            best_next = max(dp[(i+1, next_move)] for next_move in ['R', 'P', 'S'] if next_move != move)
            dp[(i, move)] = current_score + best_next

print(max(dp[(0, move)] for move in ['R', 'P', 'S']))