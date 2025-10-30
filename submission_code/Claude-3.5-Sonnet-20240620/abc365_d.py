# YOUR CODE HERE
def max_wins(N, S):
    wins = 0
    prev_move = None
    for i, move in enumerate(S):
        if move == 'R':
            if prev_move != 'P':
                wins += 1
                prev_move = 'P'
            else:
                prev_move = 'S'
        elif move == 'P':
            if prev_move != 'S':
                wins += 1
                prev_move = 'S'
            else:
                prev_move = 'R'
        else:  # move == 'S'
            if prev_move != 'R':
                wins += 1
                prev_move = 'R'
            else:
                prev_move = 'P'
    return wins

N = int(input())
S = input().strip()
print(max_wins(N, S))