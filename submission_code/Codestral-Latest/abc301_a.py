# YOUR CODE HERE
N = int(input())
S = input()

takahashi_wins = 0
aoki_wins = 0

for i in range(N):
    if S[i] == 'T':
        takahashi_wins += 1
    else:
        aoki_wins += 1

    if takahashi_wins > aoki_wins:
        winner = 'T'
    elif aoki_wins > takahashi_wins:
        winner = 'A'

print(winner)