# YOUR CODE HERE

N = int(input())
S = input()

# Initialize the count of wins
wins = 0

# Initialize the previous move of Takahashi
prev_move = ''

# Iterate over the moves of Aoki
for i in range(N):
    # If Aoki's move is Rock
    if S[i] == 'R':
        # If Takahashi's previous move was not Paper, Takahashi wins
        if prev_move != 'P':
            wins += 1
            prev_move = 'P'
    # If Aoki's move is Paper
    elif S[i] == 'P':
        # If Takahashi's previous move was not Scissors, Takahashi wins
        if prev_move != 'S':
            wins += 1
            prev_move = 'S'
    # If Aoki's move is Scissors
    else:
        # If Takahashi's previous move was not Rock, Takahashi wins
        if prev_move != 'R':
            wins += 1
            prev_move = 'R'

print(wins)