# YOUR CODE HERE

N = int(input())
S = input()

Takahashi_wins = 0
Aoki_wins = 0

for i in range(N):
    if S[i] == 'T':
        Takahashi_wins += 1
    else:
        Aoki_wins += 1

if Takahashi_wins > Aoki_wins:
    print('T')
elif Takahashi_wins < Aoki_wins:
    print('A')
else:
    print('T' if S[-1] == 'T' else 'A')