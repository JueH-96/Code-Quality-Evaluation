import sys

def max_wins(N, S):
    # Initialize variables to keep track of the previous move and the count of wins
    prev_move = ''
    win_count = 0

    # Iterate through each move in the string S
    for i in range(N):
        current_move = S[i]

        # Determine Takahashi's move based on Aoki's current move and the previous move
        if current_move == 'R':
            if prev_move != 'P':
                takahashi_move = 'P'
                win_count += 1
            else:
                takahashi_move = 'S'
        elif current_move == 'P':
            if prev_move != 'S':
                takahashi_move = 'S'
                win_count += 1
            else:
                takahashi_move = 'R'
        elif current_move == 'S':
            if prev_move != 'R':
                takahashi_move = 'R'
                win_count += 1
            else:
                takahashi_move = 'P'

        # Update the previous move
        prev_move = takahashi_move

    # Print the maximum number of games Takahashi could have won
    print(win_count)

# Read input from stdin
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

# Call the function with the input values
max_wins(N, S)