# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

# Function to determine the winning move for Takahashi
def winning_move(aoki_move):
    if aoki_move == 'R':
        return 'P'
    elif aoki_move == 'P':
        return 'S'
    elif aoki_move == 'S':
        return 'R'

# Initialize variables
takahashi_wins = 0
prev_takahashi_move = None

# Iterate through each game
for i in range(N):
    aoki_move = S[i]
    if prev_takahashi_move != winning_move(aoki_move):
        takahashi_wins += 1
        prev_takahashi_move = winning_move(aoki_move)
    else:
        prev_takahashi_move = None

print(takahashi_wins)