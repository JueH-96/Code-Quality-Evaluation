# YOUR CODE HERE
N = int(input())
S = input()

takahashi_wins = S.count('T')
aoki_wins = S.count('A')

if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    if S.index('A') < S.index('T'):
        print('A')
    else:
        print('T')