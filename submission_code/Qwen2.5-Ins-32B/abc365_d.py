import sys

def max_wins(N, S):
    # Initialize the count of wins and the previous move
    wins = 0
    prev_move = None
    
    # Iterate over the string S
    for i in range(N):
        # Determine the move that Takahashi should play
        if prev_move == 'R':
            if S[i] == 'P':
                prev_move = 'S'
            else:
                wins += 1
                prev_move = 'P'
        elif prev_move == 'P':
            if S[i] == 'S':
                prev_move = 'R'
            else:
                wins += 1
                prev_move = 'S'
        elif prev_move == 'S':
            if S[i] == 'R':
                prev_move = 'P'
            else:
                wins += 1
                prev_move = 'R'
        else:
            if S[i] == 'R':
                prev_move = 'P'
                wins += 1
            elif S[i] == 'P':
                prev_move = 'S'
                wins += 1
            else:
                prev_move = 'R'
                wins += 1
    
    return wins

# Read input from stdin
N = int(input())
S = input().strip()

# Calculate and print the maximum number of wins
print(max_wins(N, S))