# Read the input
N = int(input())
S = input()

# Initialize the maximum number of games Takahashi can win
max_wins = 0

# Initialize Takahashi's move for the first game
prev_move = None
if S[0] == 'R':
    takahashi_move = 'P'
elif S[0] == 'P':
    takahashi_move = 'S'
else:
    takahashi_move = 'R'

# Iterate through the remaining games
for i in range(1, N):
    # Determine Takahashi's move for the current game
    if S[i] == 'R':
        if takahashi_move == 'P':
            max_wins += 1
        takahashi_move = 'S'
    elif S[i] == 'P':
        if takahashi_move == 'S':
            max_wins += 1
        takahashi_move = 'R'
    else:
        if takahashi_move == 'R':
            max_wins += 1
        takahashi_move = 'P'

    # Ensure that Takahashi's move is different from the previous game
    if prev_move == takahashi_move:
        if takahashi_move == 'R':
            takahashi_move = 'P'
        elif takahashi_move == 'P':
            takahashi_move = 'S'
        else:
            takahashi_move = 'R'

    prev_move = takahashi_move

# Print the maximum number of games Takahashi can win
print(max_wins)