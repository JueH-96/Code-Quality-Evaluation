# YOUR CODE HERE
n = int(input())
s = input().strip()

# What Takahashi needs to play to win against each move
win_against = {'R': 'P', 'P': 'S', 'S': 'R'}

takahashi_moves = []
wins = 0

for i in range(n):
    aoki_move = s[i]
    winning_move = win_against[aoki_move]
    
    if i == 0:
        # First move - always try to win
        takahashi_moves.append(winning_move)
        wins += 1
    else:
        # Check if we can play the winning move (must be different from previous)
        if winning_move != takahashi_moves[i-1]:
            takahashi_moves.append(winning_move)
            wins += 1
        else:
            # Can't win, need to tie with a move different from previous
            # Find a move that ties (same as Aoki) and is different from previous
            if aoki_move != takahashi_moves[i-1]:
                takahashi_moves.append(aoki_move)
            else:
                # If tying move is same as previous, find any other move
                for move in ['R', 'P', 'S']:
                    if move != takahashi_moves[i-1]:
                        takahashi_moves.append(move)
                        break

print(wins)