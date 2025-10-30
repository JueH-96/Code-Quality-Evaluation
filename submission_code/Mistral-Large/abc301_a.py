import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

takahashi_wins = 0
aoki_wins = 0

for i in range(N):
    if S[i] == 'T':
        takahashi_wins += 1
    else:
        aoki_wins += 1

    # Check if one player has more wins than the other
    if takahashi_wins > aoki_wins + (N - i - 1):
        print('T')
        break
    elif aoki_wins > takahashi_wins + (N - i - 1):
        print('A')
        break
else:
    # If they have the same number of wins, the one who reached it first wins
    if takahashi_wins == aoki_wins:
        print('T' if S.rfind('T') > S.rfind('A') else 'A')
    else:
        print('T' if takahashi_wins > aoki_wins else 'A')